# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd.
"""CUDA-graph-safe INT8 kernels for the GLM kpool indexer.

The cache layout matches the existing FP8/INT8 paged-MQA contract: each page
stores all 128-byte quantized K rows first, followed by one FP32 scale per row.
The tail cache remains BF16 because an incomplete pool must be accumulated
before it can be Hadamard-rotated and quantized.
"""

import torch

from vllm.triton_utils import tl, triton

from vllm_metax.models.glm5next.metax.ops.kpool_compress import _hadamard128


@triton.jit
def _round_half_away_from_zero(x):
    return tl.where(x >= 0.0, tl.floor(x + 0.5), tl.ceil(x - 0.5))


@triton.jit
def _quantize_int8(x, scale):
    quantized = _round_half_away_from_zero(x / scale)
    quantized = tl.maximum(tl.minimum(quantized, 127.0), -127.0)
    return quantized.to(tl.int8)


@triton.jit
def _fwht128_quant_int8_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    input_stride_0,
    output_stride_0,
    num_rows,
):
    """Hadamard-rotate and per-row INT8-quantize one 128-wide vector."""
    row = tl.program_id(0)
    valid = row < num_rows
    offsets = tl.arange(0, 128)
    value = tl.load(
        input_ptr + row * input_stride_0 + offsets,
        mask=valid,
        other=0.0,
    ).to(tl.float32)
    # Match the FP8/BF16 path: FP32 FWHT, then materialize BF16 before quant.
    rotated = _hadamard128(value).to(tl.bfloat16).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(rotated), axis=0), 1e-4)
    scale = absmax / 127.0
    tl.store(
        output_ptr + row * output_stride_0 + offsets,
        _quantize_int8(rotated, scale),
        mask=valid,
    )
    tl.store(scale_ptr + row, scale, mask=valid)


def fwht128_quant_int8(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Hadamard-rotated INT8 Q and its per-row FP32 scale."""
    if q.ndim != 2 or q.shape[1] != 128:
        raise ValueError(f"Expected Q [rows, 128], got {q.shape}")
    if q.dtype != torch.bfloat16:
        raise TypeError(f"INT8 GLM indexer requires BF16 Q input, got {q.dtype}")
    q = q.contiguous()
    num_rows = q.shape[0]
    q_int8 = torch.empty_like(q, dtype=torch.int8)
    q_scale = torch.empty((num_rows, 1), dtype=torch.float32, device=q.device)
    if num_rows > 0:
        _fwht128_quant_int8_kernel[(num_rows,)](
            q,
            q_int8,
            q_scale,
            q.stride(0),
            q_int8.stride(0),
            num_rows,
            num_warps=1,
        )
    return q_int8, q_scale


@triton.jit
def _kpool_softmax_rotate_write_cache_int8_kernel(
    cache_int8_ptr,
    cache_fp32_ptr,
    slot_k_ptr,
    slot_score_ptr,
    ape_ptr,
    loc_ptr,
    write_mask_ptr,
    slot_k_stride_0,
    slot_k_stride_1,
    slot_score_stride_0,
    slot_score_stride_1,
    ape_stride_0,
    num_cache_blocks,
    PAGE_SIZE: tl.constexpr,
    CACHE_BYTES_PER_PAGE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SCALE_OFFSET_BYTES: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compress one candidate pool and write an INT8 K plus FP32 scale."""
    row = tl.program_id(0)
    loc = tl.load(loc_ptr + row).to(tl.int64)
    requested = tl.load(write_mask_ptr + row)
    do_write = requested & (loc >= 0) & (loc < num_cache_blocks * PAGE_SIZE)

    offsets = tl.arange(0, BLOCK_D)
    dim_mask = offsets < HEAD_DIM
    load_mask = dim_mask & do_write

    max_score = tl.full((BLOCK_D,), -float("inf"), tl.float32)
    for slot in tl.static_range(0, POOL_SIZE):
        score = tl.load(
            slot_score_ptr
            + row * slot_score_stride_0
            + slot * slot_score_stride_1
            + offsets,
            mask=load_mask,
            other=0.0,
        ).to(tl.float32)
        score += tl.load(
            ape_ptr + slot * ape_stride_0 + offsets,
            mask=load_mask,
            other=0.0,
        ).to(tl.float32)
        max_score = tl.maximum(max_score, score)

    acc = tl.full((BLOCK_D,), 0.0, tl.float32)
    denominator = tl.full((BLOCK_D,), 0.0, tl.float32)
    for slot in tl.static_range(0, POOL_SIZE):
        score = tl.load(
            slot_score_ptr
            + row * slot_score_stride_0
            + slot * slot_score_stride_1
            + offsets,
            mask=load_mask,
            other=0.0,
        ).to(tl.float32)
        score += tl.load(
            ape_ptr + slot * ape_stride_0 + offsets,
            mask=load_mask,
            other=0.0,
        ).to(tl.float32)
        probability = tl.exp(score - max_score)
        denominator += probability
        key = tl.load(
            slot_k_ptr + row * slot_k_stride_0 + slot * slot_k_stride_1 + offsets,
            mask=load_mask,
            other=0.0,
        ).to(tl.float32)
        acc += key * probability

    pooled = tl.where(do_write, acc / denominator, 0.0).to(tl.bfloat16).to(tl.float32)
    rotated = _hadamard128(pooled).to(tl.bfloat16).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(rotated), axis=0), 1e-4)
    scale = absmax / 127.0
    quantized = _quantize_int8(rotated, scale)

    page = loc // PAGE_SIZE
    page_offset = loc % PAGE_SIZE
    value_offsets = page * CACHE_BYTES_PER_PAGE + page_offset * HEAD_DIM + offsets
    scale_offset = (
        page * (CACHE_BYTES_PER_PAGE // 4) + SCALE_OFFSET_BYTES // 4 + page_offset
    )
    tl.store(cache_int8_ptr + value_offsets, quantized, mask=load_mask)
    tl.store(cache_fp32_ptr + scale_offset, scale, mask=do_write)


def kpool_compress_insert_int8(
    k: torch.Tensor,
    gate_score: torch.Tensor,
    ape: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kpool: int,
    head_dim: int,
    round_scale: bool = False,
) -> None:
    """Compress complete prefill pools into the paged INT8 index cache."""
    _validate_int8_cache(kv_cache, head_dim)
    if head_dim != 128:
        raise ValueError(f"INT8 kpool requires head_dim=128, got {head_dim}")
    if k.ndim != 2 or k.shape[1] != head_dim:
        raise ValueError(f"Expected K [tokens, {head_dim}], got {k.shape}")
    if k.dtype != torch.bfloat16 or gate_score.dtype != torch.bfloat16:
        raise TypeError("INT8 kpool requires BF16 K and gate_score inputs")
    if gate_score.shape != k.shape:
        raise ValueError("gate_score must have the same shape as K")
    if ape.shape != (kpool, head_dim) or ape.dtype != torch.float32:
        raise ValueError(f"Expected FP32 APE [{kpool}, {head_dim}], got {ape.shape}")

    num_tokens = slot_mapping.shape[0]
    if num_tokens < kpool:
        return
    positions = torch.arange(num_tokens, device=k.device)
    write_mask = (slot_mapping >= 0) & (positions >= kpool - 1)
    offsets = torch.arange(kpool, device=k.device)
    gather_indices = positions.sub(kpool - 1).clamp_min(0)[:, None] + offsets[None, :]
    slot_k = k[gather_indices].contiguous()
    slot_score = gate_score[gather_indices].contiguous()
    locations = slot_mapping.to(torch.int64).contiguous()
    write_mask = write_mask.contiguous()
    ape = ape.contiguous()

    page_size = kv_cache.shape[1]
    cache_bytes_per_page = kv_cache.stride(0)
    _kpool_softmax_rotate_write_cache_int8_kernel[(num_tokens,)](
        kv_cache.view(torch.int8),
        kv_cache.view(torch.float32),
        slot_k,
        slot_score,
        ape,
        locations,
        write_mask,
        slot_k.stride(0),
        slot_k.stride(1),
        slot_score.stride(0),
        slot_score.stride(1),
        ape.stride(0),
        kv_cache.shape[0],
        PAGE_SIZE=page_size,
        CACHE_BYTES_PER_PAGE=cache_bytes_per_page,
        POOL_SIZE=kpool,
        HEAD_DIM=head_dim,
        SCALE_OFFSET_BYTES=page_size * head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=1,
    )


@triton.jit
def _kpool_decode_update_batched_int8_kernel(
    cache_int8_ptr,
    cache_fp32_ptr,
    tail_ptr,
    tail_slot_mapping_ptr,
    key_ptr,
    key_stride_0,
    key_stride_1,
    score_ptr,
    score_stride_0,
    score_stride_1,
    ape_ptr,
    ape_stride_0,
    slot_mapping_ptr,
    positions_ptr,
    tail_stride_0,
    tail_stride_1,
    tail_stride_2,
    tail_stride_3,
    num_cache_blocks,
    num_tail_blocks,
    NEXT_N,
    PAGE_SIZE: tl.constexpr,
    CACHE_BYTES_PER_PAGE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    SCALE_OFFSET_BYTES: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Update one request's BF16 tail and emit completed INT8 pools."""
    request = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_D)
    dim_mask = offsets < HEAD_DIM

    for token_idx in tl.range(0, NEXT_N):
        flat_idx = request * NEXT_N + token_idx
        cache_loc = tl.load(slot_mapping_ptr + flat_idx).to(tl.int64)
        position = tl.load(positions_ptr + flat_idx).to(tl.int64)
        tail_slot = tl.load(tail_slot_mapping_ptr + flat_idx).to(tl.int64)

        safe_position = tl.maximum(position, 0)
        pool_slot = safe_position % POOL_SIZE
        tail_block = tl.maximum(tail_slot, 0) // POOL_SIZE
        tail_valid = (
            (tail_slot >= 0) & (tail_block >= 0) & (tail_block < num_tail_blocks)
        )
        cache_valid = (cache_loc >= 0) & (cache_loc < num_cache_blocks * PAGE_SIZE)
        completes_pool = (
            (position >= 0) & tail_valid & cache_valid & (pool_slot == POOL_SIZE - 1)
        )

        current_key = tl.load(
            key_ptr + request * key_stride_0 + token_idx * key_stride_1 + offsets,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        current_score = tl.load(
            score_ptr + request * score_stride_0 + token_idx * score_stride_1 + offsets,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)

        if completes_pool:
            pool_start = safe_position - pool_slot
            max_score = tl.full((BLOCK_D,), -float("inf"), tl.float32)
            for previous_slot in tl.static_range(0, POOL_SIZE):
                is_current = previous_slot == pool_slot
                physical_slot = (pool_start + previous_slot) % POOL_SIZE
                tail_score_offsets = (
                    tail_block * tail_stride_0
                    + tail_stride_1
                    + physical_slot * tail_stride_2
                    + offsets * tail_stride_3
                )
                previous_score = tl.load(
                    tail_ptr + tail_score_offsets,
                    mask=dim_mask & tail_valid,
                    other=0.0,
                ).to(tl.float32)
                score = tl.where(is_current, current_score, previous_score)
                score += tl.load(
                    ape_ptr + previous_slot * ape_stride_0 + offsets,
                    mask=dim_mask,
                    other=0.0,
                ).to(tl.float32)
                max_score = tl.maximum(max_score, score)

            acc = tl.full((BLOCK_D,), 0.0, tl.float32)
            denominator = tl.full((BLOCK_D,), 0.0, tl.float32)
            for previous_slot in tl.static_range(0, POOL_SIZE):
                is_current = previous_slot == pool_slot
                physical_slot = (pool_start + previous_slot) % POOL_SIZE
                tail_score_offsets = (
                    tail_block * tail_stride_0
                    + tail_stride_1
                    + physical_slot * tail_stride_2
                    + offsets * tail_stride_3
                )
                previous_score = tl.load(
                    tail_ptr + tail_score_offsets,
                    mask=dim_mask & tail_valid,
                    other=0.0,
                ).to(tl.float32)
                score = tl.where(is_current, current_score, previous_score)
                score += tl.load(
                    ape_ptr + previous_slot * ape_stride_0 + offsets,
                    mask=dim_mask,
                    other=0.0,
                ).to(tl.float32)
                probability = tl.exp(score - max_score)
                denominator += probability
                tail_key_offsets = (
                    tail_block * tail_stride_0
                    + physical_slot * tail_stride_2
                    + offsets * tail_stride_3
                )
                previous_key = tl.load(
                    tail_ptr + tail_key_offsets,
                    mask=dim_mask & tail_valid,
                    other=0.0,
                ).to(tl.float32)
                selected_key = tl.where(is_current, current_key, previous_key)
                acc += selected_key * probability

            pooled = (acc / denominator).to(tl.bfloat16).to(tl.float32)
            rotated = _hadamard128(pooled).to(tl.bfloat16).to(tl.float32)
            absmax = tl.maximum(tl.max(tl.abs(rotated), axis=0), 1e-4)
            scale = absmax / 127.0
            quantized = _quantize_int8(rotated, scale)

            page = cache_loc // PAGE_SIZE
            page_offset = cache_loc % PAGE_SIZE
            value_offsets = (
                page * CACHE_BYTES_PER_PAGE + page_offset * HEAD_DIM + offsets
            )
            scale_offset = (
                page * (CACHE_BYTES_PER_PAGE // 4)
                + SCALE_OFFSET_BYTES // 4
                + page_offset
            )
            tl.store(
                cache_int8_ptr + value_offsets,
                quantized,
                mask=dim_mask & cache_valid,
            )
            tl.store(cache_fp32_ptr + scale_offset, scale, mask=cache_valid)

        # Preserve the exact completion-before-stash ordering used by FP8/BF16.
        stash_valid = (position >= 0) & tail_valid
        tail_key_offsets = (
            tail_block * tail_stride_0
            + pool_slot * tail_stride_2
            + offsets * tail_stride_3
        )
        tail_score_offsets = tail_key_offsets + tail_stride_1
        tl.store(
            tail_ptr + tail_key_offsets,
            current_key,
            mask=dim_mask & stash_valid,
        )
        tl.store(
            tail_ptr + tail_score_offsets,
            current_score,
            mask=dim_mask & stash_valid,
        )


def kpool_decode_update_and_maybe_write_cache_int8(
    kv_cache: torch.Tensor,
    tail_kv_cache: torch.Tensor,
    tail_slot_mapping: torch.Tensor,
    key: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    pool_size: int,
    head_dim: int,
    round_scale: bool = False,  # placeholder
) -> None:
    """Batched ordered INT8 kpool update for plain/speculative decode."""
    _validate_int8_cache(kv_cache, head_dim)
    if head_dim != 128:
        raise ValueError(f"INT8 kpool requires head_dim=128, got {head_dim}")
    if tail_kv_cache.dtype != torch.bfloat16:
        raise TypeError(f"INT8 kpool requires BF16 tail, got {tail_kv_cache.dtype}")
    if key.ndim != 3 or key.shape[2] != head_dim:
        raise ValueError(f"Expected key [B, N, {head_dim}], got {key.shape}")
    if key.dtype != torch.bfloat16 or slot_score.dtype != torch.bfloat16:
        raise TypeError("INT8 kpool requires BF16 key and slot_score inputs")
    if slot_score.shape != key.shape:
        raise ValueError("slot_score must have the same shape as key")

    num_requests, next_n, _ = key.shape
    if num_requests == 0 or next_n == 0:
        return
    if tail_kv_cache.ndim != 4 or tail_kv_cache.shape[1:] != (
        2,
        pool_size,
        head_dim,
    ):
        raise ValueError(
            "Expected tail cache [num_blocks, 2, pool_size, head_dim], got "
            f"{tail_kv_cache.shape}"
        )
    expected_shape = (num_requests, next_n)
    for name, tensor in (
        ("tail_slot_mapping", tail_slot_mapping),
        ("slot_mapping", slot_mapping),
        ("positions", positions),
    ):
        if tensor.shape != expected_shape:
            raise ValueError(f"Expected {name} {expected_shape}, got {tensor.shape}")
    if ape.shape != (pool_size, head_dim) or ape.dtype != torch.float32:
        raise ValueError(
            f"Expected FP32 APE [{pool_size}, {head_dim}], got {ape.shape}"
        )

    tail_slot_mapping = tail_slot_mapping.contiguous()
    slot_mapping = slot_mapping.contiguous()
    positions = positions.contiguous()
    ape = ape.contiguous()
    page_size = kv_cache.shape[1]
    cache_bytes_per_page = kv_cache.stride(0)

    _kpool_decode_update_batched_int8_kernel[(num_requests,)](
        kv_cache.view(torch.int8),
        kv_cache.view(torch.float32),
        tail_kv_cache,
        tail_slot_mapping,
        key,
        key.stride(0),
        key.stride(1),
        slot_score,
        slot_score.stride(0),
        slot_score.stride(1),
        ape,
        ape.stride(0),
        slot_mapping,
        positions,
        tail_kv_cache.stride(0),
        tail_kv_cache.stride(1),
        tail_kv_cache.stride(2),
        tail_kv_cache.stride(3),
        kv_cache.shape[0],
        tail_kv_cache.shape[0],
        next_n,
        PAGE_SIZE=page_size,
        CACHE_BYTES_PER_PAGE=cache_bytes_per_page,
        POOL_SIZE=pool_size,
        HEAD_DIM=head_dim,
        SCALE_OFFSET_BYTES=page_size * head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=1,
    )


def _validate_int8_cache(kv_cache: torch.Tensor, head_dim: int) -> None:
    if kv_cache.dtype != torch.uint8:
        raise TypeError(f"INT8 kpool requires uint8 storage, got {kv_cache.dtype}")
    if kv_cache.ndim != 3 or kv_cache.shape[2] != head_dim + 4:
        raise ValueError(
            f"Expected INT8 cache [blocks, page, {head_dim + 4}], got {kv_cache.shape}"
        )
    if kv_cache.stride(0) % 4 != 0:
        raise ValueError("INT8 cache page stride must be divisible by 4 bytes")
