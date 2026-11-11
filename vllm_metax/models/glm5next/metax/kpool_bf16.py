# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd.
"""CUDA-graph-safe BF16 storage kernels for the GLM kpool indexer.

These kernels mirror vLLM's FP8 kpool path, but keep compressed index K in
BF16. Data-dependent cache accesses are masked inside fixed-grid Triton
kernels; no variable-length indexing tensors are created on the host.
"""

import torch

from vllm.triton_utils import tl, triton

from vllm_metax.models.glm5next.metax.ops.kpool_compress import _hadamard128


@triton.jit
def _kpool_softmax_rotate_write_cache_bf16_kernel(
    cache_ptr,
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
    cache_stride_0,
    cache_stride_1,
    cache_stride_2,
    num_cache_blocks,
    PAGE_SIZE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compress one candidate pool and write a BF16 cache entry."""
    row = tl.program_id(0)
    loc = tl.load(loc_ptr + row).to(tl.int64)
    requested = tl.load(write_mask_ptr + row)
    do_write = requested & (loc >= 0) & (loc < num_cache_blocks * PAGE_SIZE)

    offs = tl.arange(0, BLOCK_D)
    dim_mask = offs < HEAD_DIM
    load_mask = dim_mask & do_write

    max_score = tl.full((BLOCK_D,), -float("inf"), tl.float32)
    for slot in tl.static_range(0, POOL_SIZE):
        score = tl.load(
            slot_score_ptr
            + row * slot_score_stride_0
            + slot * slot_score_stride_1
            + offs,
            mask=load_mask,
            other=0.0,
        ).to(tl.float32)
        score += tl.load(
            ape_ptr + slot * ape_stride_0 + offs,
            mask=load_mask,
            other=0.0,
        ).to(tl.float32)
        max_score = tl.maximum(max_score, score)

    acc = tl.full((BLOCK_D,), 0.0, tl.float32)
    denom = tl.full((BLOCK_D,), 0.0, tl.float32)
    for slot in tl.static_range(0, POOL_SIZE):
        score = tl.load(
            slot_score_ptr
            + row * slot_score_stride_0
            + slot * slot_score_stride_1
            + offs,
            mask=load_mask,
            other=0.0,
        ).to(tl.float32)
        score += tl.load(
            ape_ptr + slot * ape_stride_0 + offs,
            mask=load_mask,
            other=0.0,
        ).to(tl.float32)
        probability = tl.exp(score - max_score)
        denom += probability
        key = tl.load(
            slot_k_ptr + row * slot_k_stride_0 + slot * slot_k_stride_1 + offs,
            mask=load_mask,
            other=0.0,
        ).to(tl.float32)
        acc += key * probability

    # Match FP8 numerical order: BF16-round before and after FWHT.
    pooled = tl.where(do_write, acc / denom, 0.0).to(tl.bfloat16).to(tl.float32)
    rotated = _hadamard128(pooled).to(tl.bfloat16)

    page = loc // PAGE_SIZE
    page_offset = loc % PAGE_SIZE
    cache_offsets = (
        page * cache_stride_0 + page_offset * cache_stride_1 + offs * cache_stride_2
    )
    tl.store(cache_ptr + cache_offsets, rotated, mask=load_mask)


def kpool_compress_insert_bf16(
    k: torch.Tensor,
    gate_score: torch.Tensor,
    ape: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kpool: int,
    head_dim: int,
    round_scale: bool = False,
) -> None:
    """Compress complete prefill pools and write them to the BF16 cache."""
    if head_dim != 128:
        raise ValueError(f"BF16 kpool requires head_dim=128, got {head_dim}")
    if kv_cache.dtype != torch.bfloat16:
        raise TypeError(f"BF16 kpool received cache dtype {kv_cache.dtype}")
    if kv_cache.ndim != 3 or kv_cache.shape[2] != head_dim:
        raise ValueError(
            f"Expected BF16 cache [blocks, page, {head_dim}], got {kv_cache.shape}"
        )
    if k.ndim != 2 or k.shape[1] != head_dim:
        raise ValueError(f"Expected K [tokens, {head_dim}], got {k.shape}")
    if k.dtype != torch.bfloat16 or gate_score.dtype != torch.bfloat16:
        raise TypeError("BF16 kpool requires BF16 K and gate_score")
    if gate_score.shape != k.shape:
        raise ValueError("gate_score must have the same shape as K")
    if ape.shape != (kpool, head_dim):
        raise ValueError(f"Expected APE [{kpool}, {head_dim}], got {ape.shape}")
    if ape.dtype != torch.float32:
        raise TypeError(f"BF16 kpool requires FP32 APE, got {ape.dtype}")

    num_tokens = slot_mapping.shape[0]
    if num_tokens < kpool:
        return

    positions = torch.arange(num_tokens, device=k.device)
    write_mask = (slot_mapping >= 0) & (positions >= kpool - 1)
    offsets = torch.arange(kpool, device=k.device)
    gather_indices = positions.sub(kpool - 1).clamp_min(0)[:, None] + offsets[None, :]
    slot_k = k[gather_indices].contiguous()
    slot_score = gate_score[gather_indices].contiguous()
    ape = ape.contiguous()
    locations = slot_mapping.to(torch.int64).contiguous()
    write_mask = write_mask.contiguous()

    _kpool_softmax_rotate_write_cache_bf16_kernel[(num_tokens,)](
        kv_cache,
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
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        kv_cache.shape[0],
        PAGE_SIZE=kv_cache.shape[1],
        POOL_SIZE=kpool,
        HEAD_DIM=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )


@triton.jit
def _gather_paged_k_bf16_kernel(
    cache_ptr,
    dst_ptr,
    block_table_ptr,
    cu_seq_lens_ptr,
    cache_stride_0,
    cache_stride_1,
    cache_stride_2,
    dst_stride_0,
    dst_stride_1,
    block_table_stride_0,
    block_table_stride_1,
    num_cache_blocks,
    batch_size,
    max_blocks,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Gather one logical token from a BF16 paged cache per program."""
    token = tl.program_id(0)
    request = tl.zeros((), tl.int32)
    sequence_offset = tl.zeros((), tl.int64)
    token_valid = tl.zeros((), tl.int1)

    # Device-side scan avoids a host synchronization on cu_seq_lens.
    for batch_idx in tl.range(0, batch_size):
        start = tl.load(cu_seq_lens_ptr + batch_idx).to(tl.int64)
        end = tl.load(cu_seq_lens_ptr + batch_idx + 1).to(tl.int64)
        in_sequence = (token >= start) & (token < end)
        request = tl.where(in_sequence, batch_idx, request)
        sequence_offset = tl.where(in_sequence, token - start, sequence_offset)
        token_valid |= in_sequence

    logical_block = sequence_offset // PAGE_SIZE
    page_offset = sequence_offset % PAGE_SIZE
    logical_valid = token_valid & (logical_block >= 0) & (logical_block < max_blocks)
    physical_block = tl.load(
        block_table_ptr
        + request * block_table_stride_0
        + logical_block * block_table_stride_1,
        mask=logical_valid,
        other=-1,
    ).to(tl.int64)
    physical_valid = (
        logical_valid & (physical_block >= 0) & (physical_block < num_cache_blocks)
    )

    offs = tl.arange(0, BLOCK_D)
    dim_mask = offs < HEAD_DIM
    cache_offsets = (
        physical_block * cache_stride_0
        + page_offset * cache_stride_1
        + offs * cache_stride_2
    )
    value = tl.load(
        cache_ptr + cache_offsets,
        mask=physical_valid & dim_mask,
        other=0.0,
    )
    tl.store(
        dst_ptr + token * dst_stride_0 + offs * dst_stride_1,
        value,
        mask=dim_mask,
    )


def gather_paged_k_bf16(
    kv_cache: torch.Tensor,
    dst_k: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
) -> None:
    """Gather a BF16 paged cache into the contiguous prefill workspace."""
    if kv_cache.dtype != torch.bfloat16 or dst_k.dtype != torch.bfloat16:
        raise TypeError("gather_paged_k_bf16 requires BF16 cache and destination")
    if kv_cache.ndim != 3 or dst_k.ndim != 2:
        raise ValueError("Expected cache [blocks, page, dim] and dst [tokens, dim]")
    if kv_cache.shape[2] != dst_k.shape[1]:
        raise ValueError(
            f"Cache head dim {kv_cache.shape[2]} does not match dst {dst_k.shape[1]}"
        )
    if block_table.ndim != 2 or cu_seq_lens.ndim != 1:
        raise ValueError(
            "Expected block_table [batch, blocks] and cu_seq_lens [batch+1]"
        )
    if cu_seq_lens.shape[0] != block_table.shape[0] + 1:
        raise ValueError("cu_seq_lens length must equal batch size + 1")

    num_tokens = dst_k.shape[0]
    batch_size = block_table.shape[0]
    if num_tokens == 0 or batch_size == 0:
        return

    _gather_paged_k_bf16_kernel[(num_tokens,)](
        kv_cache,
        dst_k,
        block_table,
        cu_seq_lens,
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        dst_k.stride(0),
        dst_k.stride(1),
        block_table.stride(0),
        block_table.stride(1),
        kv_cache.shape[0],
        batch_size,
        block_table.shape[1],
        PAGE_SIZE=kv_cache.shape[1],
        HEAD_DIM=dst_k.shape[1],
        BLOCK_D=triton.next_power_of_2(dst_k.shape[1]),
    )


@triton.jit
def _kpool_decode_update_batched_bf16_kernel(
    cache_ptr,
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
    cache_stride_0,
    cache_stride_1,
    cache_stride_2,
    tail_stride_0,
    tail_stride_1,
    tail_stride_2,
    tail_stride_3,
    num_cache_blocks,
    num_tail_blocks,
    NEXT_N,
    PAGE_SIZE: tl.constexpr,
    POOL_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Update one request's BF16 tail and emit completed pools in order."""
    request = tl.program_id(0)
    offs = tl.arange(0, BLOCK_D)
    dim_mask = offs < HEAD_DIM

    for token_idx in tl.range(0, NEXT_N):
        flat_idx = request * NEXT_N + token_idx
        cache_loc = tl.load(slot_mapping_ptr + flat_idx).to(tl.int64)
        position = tl.load(positions_ptr + flat_idx).to(tl.int64)
        tail_slot = tl.load(tail_slot_mapping_ptr + flat_idx).to(tl.int64)

        safe_position = tl.maximum(position, 0)
        pool_slot = safe_position % POOL_SIZE
        tail_block = tl.maximum(tail_slot, 0) // POOL_SIZE
        tail_block_valid = (
            (tail_slot >= 0) & (tail_block >= 0) & (tail_block < num_tail_blocks)
        )
        cache_loc_valid = (cache_loc >= 0) & (cache_loc < num_cache_blocks * PAGE_SIZE)
        completes_pool = (
            (position >= 0)
            & tail_block_valid
            & cache_loc_valid
            & (pool_slot == POOL_SIZE - 1)
        )

        current_key = tl.load(
            key_ptr + request * key_stride_0 + token_idx * key_stride_1 + offs,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)
        current_score = tl.load(
            score_ptr + request * score_stride_0 + token_idx * score_stride_1 + offs,
            mask=dim_mask,
            other=0.0,
        ).to(tl.float32)

        if completes_pool:
            pool_logical_start = safe_position - pool_slot
            max_score = tl.full((BLOCK_D,), -float("inf"), tl.float32)
            for previous_slot in tl.static_range(0, POOL_SIZE):
                is_current = previous_slot == pool_slot
                physical_slot = (pool_logical_start + previous_slot) % POOL_SIZE
                tail_score_offsets = (
                    tail_block * tail_stride_0
                    + tail_stride_1
                    + physical_slot * tail_stride_2
                    + offs * tail_stride_3
                )
                previous_score = tl.load(
                    tail_ptr + tail_score_offsets,
                    mask=dim_mask & tail_block_valid,
                    other=0.0,
                ).to(tl.float32)
                score = tl.where(is_current, current_score, previous_score)
                score += tl.load(
                    ape_ptr + previous_slot * ape_stride_0 + offs,
                    mask=dim_mask,
                    other=0.0,
                ).to(tl.float32)
                max_score = tl.maximum(max_score, score)

            acc = tl.full((BLOCK_D,), 0.0, tl.float32)
            denom = tl.full((BLOCK_D,), 0.0, tl.float32)
            for previous_slot in tl.static_range(0, POOL_SIZE):
                is_current = previous_slot == pool_slot
                physical_slot = (pool_logical_start + previous_slot) % POOL_SIZE
                tail_score_offsets = (
                    tail_block * tail_stride_0
                    + tail_stride_1
                    + physical_slot * tail_stride_2
                    + offs * tail_stride_3
                )
                previous_score = tl.load(
                    tail_ptr + tail_score_offsets,
                    mask=dim_mask & tail_block_valid,
                    other=0.0,
                ).to(tl.float32)
                score = tl.where(is_current, current_score, previous_score)
                score += tl.load(
                    ape_ptr + previous_slot * ape_stride_0 + offs,
                    mask=dim_mask,
                    other=0.0,
                ).to(tl.float32)
                probability = tl.exp(score - max_score)
                denom += probability

                tail_key_offsets = (
                    tail_block * tail_stride_0
                    + physical_slot * tail_stride_2
                    + offs * tail_stride_3
                )
                previous_key = tl.load(
                    tail_ptr + tail_key_offsets,
                    mask=dim_mask & tail_block_valid,
                    other=0.0,
                ).to(tl.float32)
                key = tl.where(is_current, current_key, previous_key)
                acc += key * probability

            pooled = (acc / denom).to(tl.bfloat16).to(tl.float32)
            rotated = _hadamard128(pooled).to(tl.bfloat16)
            cache_page = cache_loc // PAGE_SIZE
            cache_page_offset = cache_loc % PAGE_SIZE
            cache_offsets = (
                cache_page * cache_stride_0
                + cache_page_offset * cache_stride_1
                + offs * cache_stride_2
            )
            tl.store(
                cache_ptr + cache_offsets,
                rotated,
                mask=dim_mask & cache_loc_valid,
            )

        # Completion must observe previous tail entries; stash afterwards.
        stash_valid = (position >= 0) & tail_block_valid
        tail_key_offsets = (
            tail_block * tail_stride_0
            + pool_slot * tail_stride_2
            + offs * tail_stride_3
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


def kpool_decode_update_and_maybe_write_cache_bf16(
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
    """Batched, ordered BF16 kpool update for plain and speculative decode."""
    if kv_cache.dtype != torch.bfloat16:
        raise TypeError(f"BF16 kpool received cache dtype {kv_cache.dtype}")
    if tail_kv_cache.dtype != torch.bfloat16:
        raise TypeError(f"BF16 kpool received tail dtype {tail_kv_cache.dtype}")
    if head_dim != 128:
        raise ValueError(f"BF16 kpool requires head_dim=128, got {head_dim}")
    if kv_cache.ndim != 3 or kv_cache.shape[2] != head_dim:
        raise ValueError(
            f"Expected BF16 cache [blocks, page, {head_dim}], got {kv_cache.shape}"
        )
    if key.ndim != 3 or key.shape[2] != head_dim:
        raise ValueError(f"Expected key [B, N, {head_dim}], got {key.shape}")
    if key.dtype != torch.bfloat16 or slot_score.dtype != torch.bfloat16:
        raise TypeError("BF16 kpool requires BF16 key and slot_score")
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
    expected_mapping_shape = (num_requests, next_n)
    for name, tensor in (
        ("tail_slot_mapping", tail_slot_mapping),
        ("slot_mapping", slot_mapping),
        ("positions", positions),
    ):
        if tensor.shape != expected_mapping_shape:
            raise ValueError(
                f"Expected {name} {expected_mapping_shape}, got {tensor.shape}"
            )
    if ape.shape != (pool_size, head_dim):
        raise ValueError(f"Expected APE [{pool_size}, {head_dim}], got {ape.shape}")
    if ape.dtype != torch.float32:
        raise TypeError(f"BF16 kpool requires FP32 APE, got {ape.dtype}")

    tail_slot_mapping = tail_slot_mapping.contiguous()
    slot_mapping = slot_mapping.contiguous()
    positions = positions.contiguous()
    ape = ape.contiguous()

    _kpool_decode_update_batched_bf16_kernel[(num_requests,)](
        kv_cache,
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
        kv_cache.stride(0),
        kv_cache.stride(1),
        kv_cache.stride(2),
        tail_kv_cache.stride(0),
        tail_kv_cache.stride(1),
        tail_kv_cache.stride(2),
        tail_kv_cache.stride(3),
        kv_cache.shape[0],
        tail_kv_cache.shape[0],
        next_n,
        PAGE_SIZE=kv_cache.shape[1],
        POOL_SIZE=pool_size,
        HEAD_DIM=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
    )
