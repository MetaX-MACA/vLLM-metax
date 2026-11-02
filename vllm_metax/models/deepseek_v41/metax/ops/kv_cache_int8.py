# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""MetaX DeepSeek-V4.1 INT8 KV-cache support (``int8_ds_mla``).

This module owns the *layout* of the MetaX INT8 KV cache plus the
quant/dequant helpers around it. It is shared by the compressed-KV cache
(owned by the kv-source layer) and the sliding-window (SWA) cache, which use
the same block format.

Block format (``int8_ds_mla``), per paged block of ``N`` stored states
(``N`` is the number of tokens for ratio 1 and ``kernel block / ratio`` for
compressed caches)::

    +--------------------------------------------------------------+
    | state 0 values (512 B) | ... | state N-1 values (512 B)      |
    +--------------------------------------------------------------+
    | state 0 scales (64 B)  | ... | state N-1 scales (64 B)       |
    +--------------------------------------------------------------+

    values region (N * 512 B), one 512 B row per state:
      [  0 : 448)   448 x int8   NoPE values (quantized)
      [448 : 512)    64 x int8   RoPE values (quantized)
    scales region (N * 64 B), one 64 B row per state:
      16 x fp32, one per 32 dims (14 NoPE + 2 RoPE); scale k covers the value
      dims [32k, 32k + 32), i.e. dims 0..447 -> k 0..13 and RoPE -> k 14..15.

Quantization is symmetric per 32-dim group::

    scale = max(|x_block|) / INT8_MAX     (INT8_MAX = 127, floored at eps)
    q     = round(x / scale) clamped to [-127, 127]

Notes:
  * The two regions are *block-segregated*: all value bytes of the page come
    first, then all scale bytes, exactly like ``fp8_ds_mla`` segregates its
    scale bytes from the 576 B state rows. A state therefore occupies 576 B in
    total (512 value + 64 scale bytes), and ``N * 576`` is already 576-aligned,
    so the page needs no padding.
  * ``fp8_ds_mla`` is *not* byte-compatible with this layout: it keeps the 64
    RoPE dims in bf16 inside the state row and stores 8 ue8m0 scale bytes per
    state in a separate region (584 B per state). Kernels must not be shared
    between the two formats.
  * The paged tensor is int8 with shape ``[num_blocks, states_per_page, 576]``;
    the scale region is reinterpreted as fp32 inside the kernels.
"""

from __future__ import annotations

from typing import Any

import torch
import vllm_metax.envs as mx_envs

from vllm.triton_utils import tl, triton

INT8_DS_MLA_DTYPE = torch.int8
INT8_NOPE_DIM = 448
INT8_ROPE_DIM = 64
INT8_HEAD_DIM = INT8_NOPE_DIM + INT8_ROPE_DIM  # 512
INT8_QUANT_BLOCK = 32
INT8_MAX = 127.0
# Floor for the per-group scale, so an all-zero group cannot produce a zero or
# NaN dequant factor.
INT8_MIN_SCALE = 1.0 / (1 << 30)

INT8_NOPE_SCALES = INT8_NOPE_DIM // INT8_QUANT_BLOCK  # 14
INT8_ROPE_SCALES = INT8_ROPE_DIM // INT8_QUANT_BLOCK  # 2
INT8_SCALE_DIM = INT8_NOPE_SCALES + INT8_ROPE_SCALES  # 16
# Bytes one state contributes to the paged block. The page keeps them
# segregated: [states_per_page * INT8_VALUE_BYTES values][states_per_page *
# INT8_SCALE_BYTES scales], both state-major.
INT8_VALUE_BYTES = INT8_HEAD_DIM  # 512, one int8 value per element
INT8_SCALE_BYTES = INT8_SCALE_DIM * 4  # 64 = 16 x fp32
INT8_STATE_BYTES = INT8_VALUE_BYTES + INT8_SCALE_BYTES  # 576
# Historic name: the per-state stride, not the offset of the scale region.
INT8_TOKEN_BYTES = INT8_STATE_BYTES  # 576
INT8_ALIGNMENT = 576


def int8_state_byte_offsets(state_slot: int, states_per_page: int) -> tuple[int, int]:
    """Byte offsets of one state inside its page: ``(values, scales)``.

    ``state_slot`` is ``slot % states_per_page``. The values region spans
    ``[0, states_per_page * INT8_VALUE_BYTES)`` and the scales region starts
    right after it, so this mirrors what the kernels compute inline.
    """
    return (
        state_slot * INT8_VALUE_BYTES,
        states_per_page * INT8_VALUE_BYTES + state_slot * INT8_SCALE_BYTES,
    )


@triton.jit
def _int8_kv_rope_quant_store_kernel(
    kv_ptr,
    kv_stride,
    positions_ptr,
    cos_sin_cache_ptr,
    cos_sin_stride,
    k_cache_ptr,
    slot_mapping_ptr,
    k_cache_block_size,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    QUANT_BLOCK: tl.constexpr,
    VALUE_BYTES: tl.constexpr,  # 512 value bytes per state
    SCALE_BYTES: tl.constexpr,
    INT8_MAX: tl.constexpr,
    KV_BLOCK_STRIDE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
):
    """RoPE + per-32-dim INT8 quant + paged store of one KV state.

    The page is block-segregated, so this state's value bytes live at
    ``pos_in_block * VALUE_BYTES`` and its ``SCALE_BYTES`` scale bytes live in
    the scale region that starts after all ``block_size`` value rows.
    """
    token_idx = tl.program_id(0)
    slot_id = tl.load(slot_mapping_ptr + token_idx)
    if slot_id < 0:
        return

    block_idx = slot_id // k_cache_block_size
    pos_in_block = slot_id % k_cache_block_size
    page_ptr = k_cache_ptr + block_idx.to(tl.int64) * KV_BLOCK_STRIDE
    val_ptr = page_ptr + pos_in_block * VALUE_BYTES
    scale_ptr = (
        page_ptr + k_cache_block_size * VALUE_BYTES + pos_in_block * SCALE_BYTES
    ).to(tl.pointer_type(tl.float32))

    position = tl.load(positions_ptr + token_idx)
    # Compressed caches apply RoPE at the group boundary; the SWA cache uses
    # the token position directly (COMPRESS_RATIO == 1).
    if COMPRESS_RATIO > 1:
        position = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    # Pair-space view: dims (2p, 2p+1) form one GPT-J rotary pair, so the
    # trailing ROPE_DIM dims are the last ROPE_DIM // 2 pairs.
    NUM_PAIRS: tl.constexpr = HEAD_DIM // 2
    HALF_ROPE: tl.constexpr = ROPE_DIM // 2
    NOPE_PAIRS: tl.constexpr = NUM_PAIRS - HALF_ROPE
    PAIRS_PER_BLOCK: tl.constexpr = QUANT_BLOCK // 2

    offs = tl.arange(0, HEAD_DIM)
    x = tl.load(kv_ptr + token_idx * kv_stride + offs).to(tl.float32)
    even, odd = tl.split(tl.reshape(x, (NUM_PAIRS, 2)))  # each [NUM_PAIRS]
    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair_local = pair_idx - NOPE_PAIRS
    is_rope_pair = rope_pair_local >= 0
    cs_idx = tl.maximum(rope_pair_local, 0)
    cache_base = cos_sin_cache_ptr + position * cos_sin_stride
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope_pair, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope_pair, other=0.0)
    new_even = even * cos_v - odd * sin_v
    new_odd = odd * cos_v + even * sin_v

    # ---- per-QUANT_BLOCK symmetric INT8 quant (one block = 16 pairs) ----
    for blk in tl.static_range(0, HEAD_DIM // QUANT_BLOCK):
        p0 = blk * PAIRS_PER_BLOCK
        in_block = (pair_idx >= p0) & (pair_idx < p0 + PAIRS_PER_BLOCK)
        blk_even = tl.where(in_block, new_even, 0.0)
        blk_odd = tl.where(in_block, new_odd, 0.0)
        amax = tl.maximum(
            tl.max(tl.abs(blk_even), axis=0), tl.max(tl.abs(blk_odd), axis=0)
        )
        scale = tl.maximum(amax / INT8_MAX, 1.0 / (1 << 30))
        q_even = _quant_int8(blk_even / scale, INT8_MAX)
        q_odd = _quant_int8(blk_odd / scale, INT8_MAX)
        tl.store(val_ptr + 2 * pair_idx, q_even, mask=in_block)
        tl.store(val_ptr + 2 * pair_idx + 1, q_odd, mask=in_block)
        tl.store(scale_ptr + blk, scale)


@triton.jit
def _quant_int8(x, LIMIT: tl.constexpr):
    """Round-half-away-from-zero symmetric INT8 quantization."""
    rounded = x + 0.5 * tl.where(x >= 0.0, 1.0, -1.0)
    return tl.clamp(rounded, -LIMIT, LIMIT).to(tl.int8)


def quantize_and_insert_k_cache_int8(
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_dim: int = INT8_HEAD_DIM,
    rope_dim: int = INT8_ROPE_DIM,
    compress_ratio: int = 1,
) -> None:
    """RoPE + INT8 quant + paged insert into the ``int8_ds_mla`` cache.

    Args:
        kv: ``[num_tokens, head_dim]`` KV rows (RoPE *not* applied yet; the
            trailing ``rope_dim`` dims are rotated here).
        k_cache: paged int8 cache ``[num_blocks, block_size, 576]`` whose pages
            hold ``[block_size * 512 value bytes][block_size * 64 scale bytes]``.
        slot_mapping: ``[num_tokens]`` slot ids (``-1`` skips).
        positions: ``[num_tokens]`` int64 positions used for RoPE.
        cos_sin_cache: ``[max_pos, rope_dim]`` GPT-J layout (cos half, sin half).
        block_size: tokens per paged block.
        compress_ratio: group size of the target cache. ``> 1`` applies RoPE at
            the group boundary (compressed main cache); ``1`` uses the token
            position (SWA cache).
    """
    block_size = k_cache.shape[1]
    assert kv.ndim == 2 and kv.shape[1] == head_dim, (
        f"kv must be [num_tokens, {head_dim}], got {tuple(kv.shape)}"
    )
    assert k_cache.dtype == INT8_DS_MLA_DTYPE, (
        f"int8_ds_mla cache must be int8, got {k_cache.dtype}"
    )
    num_tokens = slot_mapping.shape[0]
    if num_tokens == 0:
        return
    # Byte addressing inside a page assumes the [block_size, 576] tail is
    # contiguous; only the page stride (stride(0)) may carry padding.
    assert k_cache.stride(-1) == 1 and k_cache.stride(-2) == INT8_TOKEN_BYTES, (
        "int8_ds_mla requires contiguous value/scale pages, got "
        f"strides {tuple(k_cache.stride())}"
    )
    k_cache_3d = k_cache.view(k_cache.shape[0], block_size, INT8_TOKEN_BYTES)
    _int8_kv_rope_quant_store_kernel[(num_tokens,)](
        kv,
        kv.stride(0),
        positions,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        k_cache_3d,
        slot_mapping,
        block_size,
        HEAD_DIM=head_dim,
        ROPE_DIM=rope_dim,
        QUANT_BLOCK=INT8_QUANT_BLOCK,
        VALUE_BYTES=INT8_VALUE_BYTES,
        SCALE_BYTES=INT8_SCALE_BYTES,
        INT8_MAX=INT8_MAX,
        KV_BLOCK_STRIDE=k_cache_3d.stride(0),
        COMPRESS_RATIO=compress_ratio,
        num_warps=4,
    )


def fused_deepseek_v4_kv_rope_insert_int8(
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    block_size: int,
) -> None:
    """KV-only counterpart of vLLM's ``fused_deepseek_v4_kv_rope_insert``.

    vLLM 0.32's native op intentionally accepts only packed FP8, bf16 and
    per-tensor FP8 caches. MetaX's ``int8_ds_mla`` cache uses the Triton writer
    below: GPT-J RoPE followed by symmetric INT8 quantization in groups of 32
    dims and a paged store with one fp32 scale per group.
    """
    if k_cache.shape[1] != block_size:
        raise ValueError(
            f"block_size {block_size} does not match INT8 cache page {k_cache.shape[1]}"
        )
    quantize_and_insert_k_cache_int8(
        kv,
        k_cache,
        slot_mapping,
        positions,
        cos_sin_cache,
    )


def dequantize_and_gather_k_cache_int8_ref(
    out: torch.Tensor,
    k_cache: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor | None,
    block_table: torch.Tensor,
    block_size: int,
    offset: int = 0,
) -> None:
    """Dequantize + gather a paged ``int8_ds_mla`` cache into ``out``.

    Drop-in replacement for the fp8 ``dequantize_and_gather_k_cache`` used by
    the prefill path. Like the fp8 kernel it gathers the *last* ``gather_len``
    states of each request (``start_pos = seq_len - gather_len``), because
    ``combine_topk_swa_indices`` indexes the gathered buffer relative to
    ``seq_len - gather_len``.

    TODO(metax): port to Triton/CUTEDSL. This reference implementation is
    correctness-first (it builds per-request index grids) and is meant for
    bring-up only.
    """
    num_reqs = seq_lens.shape[0]
    if num_reqs == 0:
        return
    device = k_cache.device
    assert (
        k_cache.dtype == INT8_DS_MLA_DTYPE
        and k_cache.stride(-1) == 1
        and k_cache.stride(-2) == INT8_TOKEN_BYTES
    ), (
        "int8_ds_mla gather needs an int8 cache with byte-contiguous pages, got "
        f"dtype={k_cache.dtype}, strides={tuple(k_cache.stride())}"
    )
    # One row per page: [block_size * 512 value bytes][block_size * 64 scales].
    k_cache_pages = k_cache.view(k_cache.shape[0], block_size * INT8_TOKEN_BYTES)
    lens = seq_lens if gather_lens is None else gather_lens

    value_dims = torch.arange(INT8_VALUE_BYTES, device=device, dtype=torch.int64)
    scale_dims = torch.arange(INT8_SCALE_BYTES, device=device, dtype=torch.int64)
    _, scale_region_base = int8_state_byte_offsets(0, block_size)

    for req in range(num_reqs):
        length = int(lens[req].item())
        if length <= 0:
            continue
        # The gathered window ends at seq_len (SWA passes a window shorter than
        # the sequence; the compressed cache passes gather_lens=None).
        start_pos = int(seq_lens[req].item()) - length
        pos = start_pos + torch.arange(length, device=device, dtype=torch.int64)
        blocks = block_table[req, pos // block_size].to(torch.int64)
        pos_in_block = pos % block_size
        pages = k_cache_pages[blocks]  # [length, page_bytes] int8

        value_idx = pos_in_block[:, None] * INT8_VALUE_BYTES + value_dims[None, :]
        values = pages.gather(1, value_idx).to(torch.float32)  # [length, 512]

        scale_idx = (
            scale_region_base
            + pos_in_block[:, None] * INT8_SCALE_BYTES
            + scale_dims[None, :]
        )
        scales = pages.gather(1, scale_idx).view(torch.float32)  # [length, 16]
        decoded = values * scales.repeat_interleave(INT8_QUANT_BLOCK, dim=1)
        out[req, offset : offset + length].copy_(decoded.to(out.dtype))


def _torch_ref_qnorm_rope_kv_int8_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_cache_3d: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    padded_heads: int,
    eps: float,
    block_size: int,
    rope_dim: int = INT8_ROPE_DIM,
    apply_q_norm: bool = True,
) -> torch.Tensor:
    """Reference (torch) implementation of the fused Q/KV insert for INT8.

    Mirrors ``fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert``:
      * Q: per-head RMSNorm, GPT-J RoPE, zero-filled padding head slots.
      * KV: GPT-J RoPE on the trailing ``rope_dim`` dims, per-32-dim INT8
        quant, and a paged store into the block-segregated ``int8_ds_mla``
        page: ``[block_size * 512 value bytes][block_size * 64 scale bytes]``.
    """
    head_dim = kv.shape[-1]
    nope_dim = head_dim - rope_dim
    half = rope_dim // 2
    n_heads = q.shape[1]

    cos_sin = cos_sin_cache[positions].float()  # [T, rope_dim]
    cos = cos_sin[:, :half]
    sin = cos_sin[:, half:]

    def _rope(x: torch.Tensor) -> torch.Tensor:
        nope, rot = x[..., :nope_dim], x[..., nope_dim:]
        even, odd = rot[..., 0::2], rot[..., 1::2]

        if x.dim() == 2:
            cos_b = cos
            sin_b = sin
        elif x.dim() == 3:
            cos_b = cos.unsqueeze(1)
            sin_b = sin.unsqueeze(1)
        else:
            raise ValueError(f"unexpected x.dim()={x.dim()}")

        rot = torch.stack(
            (even * cos_b - odd * sin_b, even * sin_b + odd * cos_b), dim=-1
        ).flatten(-2)
        return torch.cat((nope, rot), dim=-1)

    # ---- Q: optional per-head RMSNorm, RoPE and head padding ----
    q_out = torch.zeros(
        (q.shape[0], padded_heads, head_dim), dtype=q.dtype, device=q.device
    )
    q_float = q.float()
    if apply_q_norm:
        q_float = q_float * torch.rsqrt(
            q_float.square().mean(dim=-1, keepdim=True) + eps
        )
    q_out[:, :n_heads] = _rope(q_float).to(q.dtype)

    # ---- KV: RoPE + INT8 quant + paged store (block-segregated page) ----
    num_tokens = slot_mapping.shape[0]
    if num_tokens > 0:
        kv_roped = _rope(kv.float()[:num_tokens])
        grouped = kv_roped.view(num_tokens, INT8_SCALE_DIM, INT8_QUANT_BLOCK)
        scales = (grouped.abs().amax(dim=-1) / INT8_MAX).clamp_min(INT8_MIN_SCALE)
        scaled = grouped / scales.unsqueeze(-1)
        q_int8 = (
            torch.where(scaled >= 0, (scaled + 0.5).floor(), (scaled - 0.5).ceil())
            .clamp(-INT8_MAX, INT8_MAX)
            .to(torch.int8)
        )
        valid = slot_mapping >= 0
        blocks = (slot_mapping // block_size)[valid]
        pos_in_block = (slot_mapping % block_size)[valid]
        pages = k_cache_3d.view(k_cache_3d.shape[0], block_size * INT8_TOKEN_BYTES)
        value_dims = torch.arange(
            INT8_VALUE_BYTES, device=q_int8.device, dtype=torch.int64
        )
        value_idx = pos_in_block[:, None] * INT8_VALUE_BYTES + value_dims[None, :]
        # Advanced indexing broadcasts the index tensors right-aligned, so the
        # block index needs an explicit trailing axis to pair with value_idx.
        # `q_int8` is [num_tokens, INT8_SCALE_DIM, INT8_QUANT_BLOCK]; flatten it
        # to one 512 byte row per state.
        pages[blocks[:, None], value_idx] = q_int8[valid].view(-1, INT8_VALUE_BYTES)
        scale_dims = torch.arange(
            INT8_SCALE_BYTES, device=q_int8.device, dtype=torch.int64
        )
        _, scale_region_base = int8_state_byte_offsets(0, block_size)
        scale_idx = (
            scale_region_base
            + pos_in_block[:, None] * INT8_SCALE_BYTES
            + scale_dims[None, :]
        )
        pages[blocks[:, None], scale_idx] = (
            scales[valid].contiguous().view(torch.int8).view(-1, INT8_SCALE_BYTES)
        )
    return q_out


def fused_qnorm_rope_kv_int8_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    padded_heads: int,
    eps: float,
    block_size: int,
    apply_q_norm: bool = True,
) -> torch.Tensor:
    """
    Q(norm+RoPE) + KV(RoPE+INT8 quant+insert) for the ``int8_ds_mla`` cache.
    """
    if not mx_envs.USE_PRECOMPILED_KERNEL:
        # NOTE(metax): The fused MCOPLIB entry is absent from source builds.
        # Keep Q math on device and use the existing native Triton KV writer;
        # the bring-up reference's boolean indexing is not graph safe.
        if q.ndim != 3 or q.shape[-1] != INT8_HEAD_DIM:
            raise ValueError("INT8 Q must have shape [tokens, heads, 512]")
        if padded_heads != 0 and padded_heads < q.shape[1]:
            raise ValueError("padded_heads must cover all Q heads")
        if k_cache.shape[1] != block_size:
            raise ValueError("block_size must match the INT8 cache page")
        q_out = torch.zeros(
            (q.shape[0], padded_heads, INT8_HEAD_DIM), dtype=q.dtype, device=q.device
        )
        if padded_heads:
            q_float = q.float()
            if apply_q_norm:
                q_float = q_float * torch.rsqrt(
                    q_float.square().mean(dim=-1, keepdim=True) + eps
                )
            cos_sin = cos_sin_cache[positions].float()
            cos, sin = cos_sin.chunk(2, dim=-1)
            cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
            nope = q_float[..., :INT8_NOPE_DIM]
            rotary = q_float[..., INT8_NOPE_DIM:]
            even, odd = rotary[..., 0::2], rotary[..., 1::2]
            rotated = torch.stack(
                (even * cos - odd * sin, even * sin + odd * cos), dim=-1
            ).flatten(-2)
            q_out[:, : q.shape[1]].copy_(torch.cat((nope, rotated), dim=-1).to(q.dtype))
        quantize_and_insert_k_cache_int8(
            kv, k_cache, slot_mapping, positions, cos_sin_cache
        )
        return q_out
    return torch.ops._C.fused_deepseek_v41_qnorm_rope_kv_rope_int8_insert(  # type: ignore[attr-defined]
        q,
        kv,
        k_cache,
        slot_mapping,
        positions,
        cos_sin_cache,
        padded_heads,
        eps,
        block_size,
        apply_q_norm,
    )


def int8_paged_decode_attention(*args: Any, **kwargs: Any) -> torch.Tensor:
    """Stub: paged INT8 decode attention kernel.

    Decode reads the SWA/compressed cache directly (no dequant-gather), so it
    needs an INT8 variant of the FlashMLA / FlashInfer paged decode kernel.
    It does not exist on MetaX yet, so fail loudly instead of returning
    garbage.

    TODO(metax): implement the ``int8_ds_mla`` (per-32-dim fp32 scales) paged
    decode kernel and wire it into
    ``DeepseekV4FlashMLAAttention._forward_decode``.
    """
    raise NotImplementedError(
        "int8_ds_mla decode attention is not implemented on MetaX yet. Use "
        "--kv-cache-dtype fp8_ds_mla for decode, or implement the INT8 paged "
        "decode kernel (vllm_metax/models/deepseek_v4_1/kv_cache_int8.py)."
    )


@triton.jit
def _indexer_k_norm_rope_quant_store_int8_kernel(
    k_pre_ptr,
    k_pre_stride,
    positions_ptr,
    rms_norm_weight_ptr,
    rms_norm_eps,
    cos_sin_cache_ptr,
    cos_sin_stride,
    k_cache_ptr,
    kv_slot_mapping_ptr,
    kv_cache_block_size,
    HEAD_SIZE: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    KV_BLOCK_STRIDE: tl.constexpr,
    INT8_MAX: tl.constexpr,
):
    """k_norm -> GPT-J RoPE -> INT8 quant -> paged store (one token/program)."""
    token_idx = tl.program_id(0)

    kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx)
    if kv_slot_idx < 0:
        return
    position = tl.load(positions_ptr + token_idx)
    # Only the last token of a group publishes that group's index key.
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    block = tl.arange(0, HEAD_SIZE)

    # ---- k_norm (fp32, with the bf16 roundtrip the reference uses) ----
    k = tl.load(k_pre_ptr + token_idx * k_pre_stride + block).to(tl.float32)
    rms_w = tl.load(rms_norm_weight_ptr + block).to(tl.float32)
    variance = tl.sum(k * k, axis=0) / HEAD_SIZE
    k = (k * tl.rsqrt(variance + rms_norm_eps) * rms_w).to(tl.bfloat16).to(tl.float32)

    # ---- register-based GPT-J forward RoPE ----
    # A latent stands for the first token of its group, so group j takes
    # position j * compress_ratio.
    NUM_PAIRS: tl.constexpr = HEAD_SIZE // 2
    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM
    NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2

    even, odd = tl.split(tl.reshape(k, (NUM_PAIRS, 2)))
    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair_local = pair_idx - NOPE_PAIRS
    is_rope_pair = rope_pair_local >= 0
    cs_idx = tl.maximum(rope_pair_local, 0)

    compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    cache_base = cos_sin_cache_ptr + compressed_pos * cos_sin_stride
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope_pair, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope_pair, other=0.0)

    new_even = even * cos_v - odd * sin_v
    new_odd = odd * cos_v + even * sin_v
    new_even = new_even.to(tl.bfloat16).to(tl.float32)
    new_odd = new_odd.to(tl.bfloat16).to(tl.float32)

    # ---- paged cache pointers (segregated: values first, then scales) ----
    kv_block_idx = kv_slot_idx // kv_cache_block_size
    kv_pos_in_block = kv_slot_idx % kv_cache_block_size
    cache_block_ptr = k_cache_ptr + kv_block_idx.to(tl.int64) * KV_BLOCK_STRIDE
    val_ptr = cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE
    scale_ptr = (
        cache_block_ptr
        + kv_cache_block_size * TOKEN_STRIDE
        + kv_pos_in_block * SCALE_DIM
    ).to(tl.pointer_type(tl.float32))

    # ---- per-token symmetric INT8 quant (single 128-wide block) ----
    result = tl.interleave(new_even, new_odd)  # [HEAD_SIZE] fp32
    result_bf16 = result.to(tl.bfloat16).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(result_bf16), axis=0), 1e-4)
    scale = absmax / INT8_MAX
    # Round half away from zero; the int8 cast truncates toward zero, so
    # adding +/-0.5 first gives the CUDA reference's rounding.
    scaled = result_bf16 / scale
    rounded = scaled + 0.5 * tl.where(scaled >= 0.0, 1.0, -1.0)
    x_int8 = tl.clamp(rounded, -INT8_MAX, INT8_MAX).to(tl.int8)
    tl.store(val_ptr + tl.arange(0, TOKEN_STRIDE), x_int8)
    tl.store(scale_ptr, scale)


def indexer_k_norm_rope_store_ref(
    k_pre: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    k_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    compress_ratio: int,
) -> None:
    """INT8 variant of ``indexer_k_norm_rope_store``.

    Args:
        k_pre: ``[num_tokens, 128]`` bf16 ``wk(latent)`` projection; only
            group-boundary rows are read.
        positions: ``[num_tokens]`` int64 positions.
        cos_sin_cache: ``[max_pos, rope_head_dim]`` GPT-J layout.
        rms_norm_weight: ``[128]`` k_norm weight.
        k_cache: int8 paged indexer cache ``[num_blocks, block_size, row_bytes]``.
        kv_slot_mapping: ``[num_tokens]`` slots in the indexer cache
            (``-1`` = skip).
        compress_ratio: group size; keys are emitted at group boundaries.
    """
    num_tokens = kv_slot_mapping.numel()
    if num_tokens == 0:
        return
    assert k_pre.ndim == 2 and k_pre.shape[1] == 128
    assert k_pre.dtype == torch.bfloat16 and k_pre.stride(1) == 1
    assert k_cache.dtype == torch.int8, (
        f"INT8 indexer cache must be int8, got {k_cache.dtype}"
    )
    assert compress_ratio in (1, 2)

    head_dim = k_pre.shape[1]
    token_stride = head_dim  # 128
    scale_dim = 4  # one fp32 scale per stored token

    _indexer_k_norm_rope_quant_store_int8_kernel[(num_tokens,)](
        k_pre,
        k_pre.stride(0),
        positions,
        rms_norm_weight,
        rms_norm_eps,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        k_cache,
        kv_slot_mapping,
        k_cache.shape[1],
        HEAD_SIZE=head_dim,
        ROPE_HEAD_DIM=64,
        COMPRESS_RATIO=compress_ratio,
        TOKEN_STRIDE=token_stride,
        SCALE_DIM=scale_dim,
        KV_BLOCK_STRIDE=k_cache.stride(0),
        INT8_MAX=INT8_MAX,
        num_warps=1,
    )


""" from mcoplib """

_ON_GFX950 = False


@triton.jit
def _rope_quant_insert_kernel(
    latent,
    positions,
    cos_sin,
    cache,
    cache_slots,
    NUM_TOKENS,
    COS_STRIDE: tl.constexpr,
    CACHE_STRIDE: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    SANITIZE_CACHE_NANS: tl.constexpr,
    BT: tl.constexpr = 1,
):
    """Paged writer, 2D-tiled: one program handles ``BT`` tokens as a ``[BT, 512]``
    tile so the scalar slot/position/cos-sin loads amortize across the tile and the
    latent load/cache store vectorize per row.

    ``QUANT_TYPE`` selects the layout (both segregated per page: values region then
    scales region):

    * fp8 (QUANT_TYPE=0) -- 584 bytes/token: 576 value bytes
      ([448 NoPE fp8(e4m3) | 128 RoPE bf16]) then 8 scale bytes (7 UE8M0 per-group
      exponents over the NoPE dims + 1 pad). The RoPE payload is stored as bf16.
    * int8 (QUANT_TYPE=1) -- 576 bytes/token: 512 value bytes (the FULL rotated row
      quantized to int8: 448 NoPE identity dims | 64 GPT-J-rotated RoPE dims) then
      64 scale bytes (16 fp32 per-group scales, one per 32 dims). RoPE is rotated
      first, then quantized alongside the NoPE dims (groups 14/15).
    """
    pid = tl.program_id(0)
    rows = pid * BT + tl.arange(0, BT)  # token ids [BT]
    m = rows < NUM_TOKENS
    d = tl.arange(0, 512)
    j = tl.arange(0, 32)

    slot = tl.load(cache_slots + rows, m, other=-1)
    position = tl.load(positions + rows, m, other=0)
    valid = m & (slot >= 0) & ((position + 1) % COMPRESS_RATIO == 0)
    vmask = valid[:, None]

    base = rows[:, None].to(tl.int64) * 512
    normed = tl.load(latent + base + d[None, :], vmask).to(tl.float32)  # [BT,512]
    page = (slot // CACHE_BLOCK).to(tl.int64) * CACHE_STRIDE
    slot_in = (slot % CACHE_BLOCK)[:, None]

    values = page[:, None] + slot_in * 512  # [BT,1]
    scales = page[:, None] + CACHE_BLOCK * 512 + slot_in * 64
    # quantize the whole row as 16 groups; only groups 0..13 (dims 0..447)
    # are stored/kept -- groups 14/15 here are provisional (unrotated) and are
    g16 = tl.reshape(normed, (BT, 16, 32))
    amaxN = tl.maximum(tl.max(tl.abs(g16), 2), 1e-4)  # [BT,16]
    scaleN = amaxN * (1.0 / 127.0)
    qN = tl.extra.cuda.libdevice.round(g16 * tl.reshape(127.0 / amaxN, (BT, 16, 1)))
    qN = tl.clamp(qN, -127.0, 127.0).to(tl.int8)
    qN = tl.reshape(qN, (BT, 512)).to(tl.uint8, bitcast=True)
    tl.store(cache + values + d[None, :], qN, vmask & (d[None, :] < 448))
    # RoPE tail: load the 64 dims, GPT-J rotate (32 interleave pairs), quantize
    # as 2 groups of 32.
    tail = tl.load(latent + base + 448 + tl.arange(0, 64)[None, :], vmask).to(
        tl.float32
    )
    te, to = tl.split(tl.reshape(tail, (BT, 32, 2)))
    cs = (
        cos_sin
        + (position // COMPRESS_RATIO * COMPRESS_RATIO)[:, None].to(tl.int64)
        * COS_STRIDE
    )
    c = tl.load(cs + j[None, :], vmask, other=1.0).to(tl.float32)  # [BT,32]
    sn = tl.load(cs + 32 + j[None, :], vmask, other=0.0).to(tl.float32)
    rot = tl.interleave(te * c - to * sn, to * c + te * sn)  # [BT,64]
    if SANITIZE_CACHE_NANS:
        rot = tl.where(rot == rot, rot, 0.0)
    gR = tl.reshape(rot, (BT, 2, 32))
    amaxR = tl.maximum(tl.max(tl.abs(gR), 2), 1e-4)  # [BT,2]
    scaleR = amaxR * (1.0 / 127.0)
    qR = tl.extra.cuda.libdevice.round(gR * tl.reshape(127.0 / amaxR, (BT, 2, 1)))
    qR = tl.clamp(qR, -127.0, 127.0).to(tl.int8)
    qR = tl.reshape(qR, (BT, 64)).to(tl.uint8, bitcast=True)
    tl.store(cache + values + 448 + tl.arange(0, 64)[None, :], qR, vmask)
    # 16 fp32 scales: groups 0..13 from the NoPE quant, 14..15 from the rotated
    # tail quant, contiguous in the scale region.
    sc_f32 = (cache + scales).to(tl.pointer_type(tl.float32))
    g16i = tl.arange(0, 16)[None, :]
    tl.store(sc_f32 + g16i, scaleN, vmask & (g16i < 14))
    tl.store(sc_f32 + 14 + tl.arange(0, 2)[None, :], scaleR, vmask)


def rope_quant_insert_int8(
    latent: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    compress_ratio: int,
) -> None:
    assert compress_ratio in (1, 2)
    assert latent.shape[1] == 512 and latent.dtype == torch.bfloat16
    assert latent.is_contiguous()
    num_tokens = slot_mapping.numel()
    assert num_tokens <= min(latent.shape[0], positions.numel())
    if num_tokens == 0:
        return

    is_paged = kv_cache.dtype == torch.uint8 or (
        kv_cache.dtype == torch.int8 and kv_cache.shape[-1] in (576, 584)
    )
    is_int8 = kv_cache.dtype == torch.int8

    assert is_int8

    def _bt(n, int8, paged):
        if n < 16:
            return 1
        if int8:
            # per-token full-row reduction + libdevice.round dominate; small tiles
            # keep the reduction cheap and avoid register spilling.
            if paged:
                # tail-only rotate + two-store quant: register pressure peaks the
                # [BT,512] load/store at BT=4 for large T (measured 443 GB/s), then
                # spills past BT=4 (BT=8=343, BT=16=204). Small T stays narrow.
                if n >= 2048:
                    return 4
                if n >= 256:
                    return 2
                return 1
            return 4 if n >= 256 else 2
        # bf16 identity / fp8 per-group: coalesced streaming, tolerate wider tiles.
        if n >= 2048:
            return 8
        if n >= 256:
            return 4
        return 2

    bt = _bt(num_tokens, is_int8, is_paged)
    grid = ((num_tokens + bt - 1) // bt,)

    # ---- paged layouts, segregated value+scale region ----
    # int8 is shared with the plain path, so disambiguate paged vs plain by the last
    # dim: fp8 paged -> 584, int8 paged -> 576, plain row -> 512.
    if is_paged:
        assert kv_cache.shape[-1] == 576
        _rope_quant_insert_kernel[grid](
            latent,
            positions,
            cos_sin_cache,
            kv_cache,
            slot_mapping,
            num_tokens,
            COS_STRIDE=cos_sin_cache.stride(0),
            CACHE_STRIDE=kv_cache.stride(0),
            CACHE_BLOCK=kv_cache.shape[1],
            COMPRESS_RATIO=compress_ratio,
            SANITIZE_CACHE_NANS=_ON_GFX950,
            BT=bt,
            num_warps=1,
        )
        return


@triton.jit
def _dequantize_and_gather_int8_k_kernel(
    out_ptr,
    out_stride0,
    out_stride1,
    k_cache_ptr,
    seq_lens_ptr,
    block_table_ptr,
    offset,
    gather_lens_ptr,
    # Constants
    max_blocks_per_seq: tl.constexpr,
    scale_dim: tl.constexpr,  # 64
    quant_block: tl.constexpr,  # 32 (quantization block size)
    cache_block_size: tl.constexpr,  # 64 or 128 (paged cache block size)
    token_data_size: tl.constexpr,  # 512 bytes per token data
    block_stride: tl.constexpr,  # total bytes per block (padded) int32
    output_dim: tl.constexpr,  # 512
    n_quant_blocks: tl.constexpr,  # 16 real blocks
):
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    if gather_lens_ptr is not None:  # noqa: SIM108
        gather_len = tl.load(gather_lens_ptr + batch_idx)
    else:
        # Gather all tokens
        gather_len = seq_len
    start_pos = seq_len - gather_len

    for i in range(worker_id, gather_len, num_workers):
        # Calculate the actual token index in the sequence
        pos = start_pos + i

        # Calculate which block and position within block
        block_in_seq = pos // cache_block_size
        pos_in_block = pos % cache_block_size

        # Get physical block index from block table
        block_table_row_ptr = block_table_ptr + batch_idx * max_blocks_per_seq
        physical_block_idx = tl.load(block_table_row_ptr + block_in_seq)  # int32

        # int64: physical_block_idx * block_stride can exceed 2^31 with many
        # KV-cache blocks (e.g. >= 57K at block_stride ~37K).
        cache_block_ptr = k_cache_ptr + physical_block_idx.to(tl.int64) * block_stride

        # Token data pointer
        token_data_ptr = cache_block_ptr + pos_in_block * token_data_size

        # Scale pointer: after all token data
        token_scale_ptr = (
            cache_block_ptr
            + cache_block_size * token_data_size
            + pos_in_block * scale_dim
        )

        output_row_ptr = out_ptr + batch_idx * out_stride0 + (offset + i) * out_stride1
        for qblock_idx in tl.static_range(n_quant_blocks):
            qblock_start = qblock_idx * quant_block
            offsets = qblock_start + tl.arange(0, quant_block)
            mask = offsets < output_dim
            x_int8 = tl.load(token_data_ptr + offsets, mask=mask, other=0).to(
                tl.int8, bitcast=True
            )
            x_float = x_int8.to(tl.float32)
            scale_ptr_f32 = token_scale_ptr.to(tl.pointer_type(tl.float32))
            scale = tl.load(scale_ptr_f32 + qblock_idx)
            x_dequant = x_float * scale
            tl.store(output_row_ptr + offsets, x_dequant.to(tl.bfloat16), mask=mask)


def dequantize_and_gather_k_cache_triton_int8(
    # [num_reqs, max_num_tokens, head_size]
    out: torch.Tensor,
    # [num_blocks, block_size, head_bytes]
    k_cache: torch.Tensor,
    # [num_reqs]
    seq_lens: torch.Tensor,
    # [num_reqs]
    gather_lens: torch.Tensor | None,
    # [num_reqs, max_blocks_per_seq]
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    num_reqs = seq_lens.shape[0]
    NUM_WORKERS = 128
    TOKEN_DATA_SIZE = 512
    TOKEN_SCALE_DIM = 64
    QUANT_BLOCK_SIZE = 32
    N_QUANT_BLOCKS = 16

    _dequantize_and_gather_int8_k_kernel[(num_reqs, NUM_WORKERS)](
        out,
        out.stride(0),
        out.stride(1),
        k_cache,
        seq_lens,
        block_table,
        offset,
        gather_lens,
        max_blocks_per_seq=block_table.shape[-1],
        scale_dim=TOKEN_SCALE_DIM,
        quant_block=QUANT_BLOCK_SIZE,
        cache_block_size=block_size,
        token_data_size=TOKEN_DATA_SIZE,
        block_stride=k_cache.stride(0),
        output_dim=512,
        n_quant_blocks=N_QUANT_BLOCKS,
    )


def dequantize_and_gather_k_cache_int8(
    # [num_reqs, max_num_tokens, head_size]
    out: torch.Tensor,
    # [num_blocks, block_size, head_bytes]
    k_cache: torch.Tensor,
    # [num_reqs]
    seq_lens: torch.Tensor,
    # [num_reqs]
    gather_lens: torch.Tensor | None,
    # [num_reqs, max_blocks_per_seq]
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    dequantize_and_gather_k_cache_triton_int8(
        out,
        k_cache,
        seq_lens,
        gather_lens,
        block_table,
        block_size,
        offset,
    )


"""Indexer K production for DeepSeek V4.1 kv-source layers.

In v4.1 the index key is derived from the *main* compressor's latent:
``k = k_norm(wk(latent))`` (reference model.py Indexer.forward), then RoPE'd
at the group's first-token position and MXFP4/FP8/INT8-quantized into the
paged indexer K cache. The ``wk(latent)`` GEMM runs in torch; this kernel
fuses the remaining k_norm → RoPE → quant → paged store.

Unlike the legacy (v4.0) indexer path there is no per-token pooling from a
compressor state cache: the latent already stands for a whole group, so only
group-boundary tokens ``(position + 1) % compress_ratio == 0`` produce a key.

Quant modes (``QUANT_MODE`` constexpr, fully decoupled code paths):

* ``0`` — per-token FP8 (e4m3) with a single fp32 power-of-2 scale.
* ``1`` — MXFP4 (2 E2M1 nibbles/byte + ue8m0 per 32 elements).
* ``2`` — per-token INT8: the whole 128-dim head is one quant block carrying
  a single fp32 scale ``amax / 127`` (dequant = ``code * scale``); its 4
  bytes live in the block's scale region after the value bytes — the same
  segregated per-token layout the FP8 mode uses.
"""

import os


@triton.jit
def _fp32x2_to_fp4x2(x_lo, x_hi):
    # NOTE: $1 is high nibble, $2 is low nibble
    return tl.inline_asm_elementwise(
        """
        {
            .reg .b8 tmp;
            cvt.rn.satfinite.e2m1x2.f32 tmp, $1, $2;
            cvt.u32.u8 $0, tmp;
        }
        """,
        constraints="=r,f,f",
        args=[x_hi, x_lo],
        dtype=tl.uint32,
        is_pure=True,
        pack=1,
    ).to(tl.uint8)


@triton.jit
def _fp32_to_e2m1_nibble(x):
    # E2M1 round-to-nearest-even via the reference quantizer's boundaries:
    #   values    [0.00, 0.50, 1.00, 1.50, 2.00, 3.00, 4.00, 6.00]
    #   boundaries [0.25, 0.75, 1.25, 1.75, 2.50, 3.50, 5.00]
    # Bit-identical to cvt.rn.satfinite.e2m1x2.f32 on values in [-6, 6].
    ax = tl.abs(x)
    code = tl.where(ax > 0.25, 1, 0)
    code = tl.where(ax >= 0.75, 2, code)
    code = tl.where(ax > 1.25, 3, code)
    code = tl.where(ax >= 1.75, 4, code)
    code = tl.where(ax > 2.5, 5, code)
    code = tl.where(ax >= 3.5, 6, code)
    code = tl.where(ax > 5.0, 7, code)
    sign = tl.where(x < 0.0, 8, 0)
    return (code + sign).to(tl.uint8)


@triton.jit
def _fp32x2_to_fp4x2_sw(x_lo, x_hi):
    # Software MXFP4 pack: even index -> low nibble, odd index -> high nibble
    # (same convention as the PTX path above).
    return _fp32_to_e2m1_nibble(x_lo) | (_fp32_to_e2m1_nibble(x_hi) << 4)


# MXFP4 value layout: one quant block per 32 elements
# (16 packed e2m1 bytes + 1 ue8m0 scale byte).
MXFP4_BLOCK_SIZE = 32

# Quant modes for _indexer_k_norm_rope_quant_store_kernel.
_QUANT_FP8 = 0
_QUANT_FP4 = 1
_QUANT_INT8 = 2

# fp32 magic number for round-to-nearest-even integer conversion:
# adding 1.5 * 2**23 forces the fp32 adder to round to the nearest integer.
_RNE_MAGIC = 12582912.0  # 1.5 * 2**23
_RNE_MAGIC_BITS = 0x4B400000


def _fp4_asm_default() -> bool:
    """Pick the MXFP4 pack implementation.

    MACA (triton 3.6.0 / maca 3.8.x) cannot even load a kernel image that
    contains ``cvt.rn.satfinite.e2m1x2`` — the specialization fails at
    ``load_binary`` with "device kernel image is invalid". Default to the
    bit-identical software pack there; keep the PTX path everywhere else.
    ``MCOP_INDEXER_K_FP4_ASM=1|0`` overrides the detection for debugging.
    """
    if "MCOP_INDEXER_K_FP4_ASM" in os.environ:
        return os.environ["MCOP_INDEXER_K_FP4_ASM"] == "1"
    return "metax" not in torch.__version__.lower()


# Resolved once at import: the launcher sits on the critical path of every
# decode step, and per-call re-detection is measurable host overhead once
# kernel time drops to a few microseconds.
_FP4_ASM_DEFAULT = _fp4_asm_default()


def indexer_k_norm_rope_store_int8(
    k_pre: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    k_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    compress_ratio: int,
    use_fp4_cache: bool = False,
    use_int8_cache: bool = True,
) -> None:
    """k_norm → RoPE → quant → paged store for indexer keys.

    Args:
        k_pre: [num_tokens, 128] bf16, the ``wk(latent)`` projection. Only
            group-boundary rows are read.
        positions: [num_tokens] int64 token positions. Positions of rows
            below ``num_tokens`` must be inside ``cos_sin_cache`` (the table
            spans the model's whole position range); rows past
            ``num_tokens`` may hold anything — they are never dereferenced.
        cos_sin_cache: [max_pos, rope_head_dim] GPT-J layout (cos half, then
            sin half), from the layer's compress-RoPE instance.
        rms_norm_weight: [128] k_norm weight.
        k_cache: uint8 paged indexer cache [num_blocks, block_size, row_bytes]
            with the values-then-scales layout: token ``s`` of block ``b``
            lives at flat offsets ``b * stride0 + (s // 1) * token_stride`` for
            its ``token_stride`` value bytes and
            ``b * stride0 + block_size * token_stride + s * scale_dim`` for its
            ``scale_dim`` scale bytes.
        kv_slot_mapping: [num_tokens] slots in the indexer cache (-1 = skip).
        compress_ratio: group size; keys are emitted at group boundaries.
        use_fp4_cache: MXFP4 (2 nibbles/byte + ue8m0 per 32) when True, else
            per-token FP8 with a single fp32 scale.
        use_int8_cache: per-token INT8 (one fp32 scale ``scale = amax / 127``
            for the whole 128-dim head, stored as 4 bytes in the scale
            region) when True. Mutually exclusive with ``use_fp4_cache``.
    """
    num_tokens = kv_slot_mapping.numel()
    assert k_pre.ndim == 2 and k_pre.shape[1] == 128
    assert k_pre.dtype == torch.bfloat16 and k_pre.stride(1) == 1
    assert num_tokens <= k_pre.shape[0] and num_tokens <= positions.numel()
    assert compress_ratio in (1, 2)
    assert cos_sin_cache.stride(-1) == 1, "cos_sin_cache rows must be contiguous"
    assert k_cache.stride(-1) == 1, "k_cache rows must be contiguous"
    assert not (use_int8_cache and use_fp4_cache), (
        "use_int8_cache and use_fp4_cache are mutually exclusive"
    )
    if num_tokens == 0:
        return

    head_dim = k_pre.shape[1]
    if use_int8_cache:
        token_stride = head_dim
        scale_dim = 4  # single fp32 scale per token
        quant_mode = _QUANT_INT8
    elif use_fp4_cache:
        token_stride = head_dim // 2
        scale_dim = head_dim // MXFP4_BLOCK_SIZE
        quant_mode = _QUANT_FP4
    else:
        token_stride = head_dim
        scale_dim = 4  # single float32 scale
        quant_mode = _QUANT_FP8

    # Value layout inside each cache block is linear row-major: the MACA
    # gather reader (cp_gather_indexer_k_quant_cache_triton) selects it for
    # every block size. The 16x16-tiled ("SHUFFLE") layout that ROCm's
    # deepgemm_fp8_paged_mqa_logits(Preshuffle=True) reader expects is not
    # provided by this build.

    # Tokens per program: keep >= 32 bytes of k_pre per thread on C600-U
    # (64 lanes/warp) once there is enough parallelism to fill the APs;
    # stay at one token per program for small batches so the grid stays wide.
    if num_tokens >= 2048:
        tp = 8
    elif num_tokens >= 256:
        tp = 4
    else:
        tp = 1

    # Keep the launch path lean: with a ~5 us kernel every host-side
    # microsecond is percent-level wall time, and triton's generated binder
    # pays for every parameter in the kernel signature on each call.
    _indexer_k_norm_rope_quant_store_kernel[((num_tokens + tp - 1) // tp,)](
        k_pre,
        k_pre.stride(0),
        positions,
        rms_norm_weight,
        rms_norm_eps,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        k_cache,
        kv_slot_mapping,
        num_tokens,
        k_cache.shape[1],
        HEAD_SIZE=head_dim,
        ROPE_HEAD_DIM=64,
        COMPRESS_RATIO=compress_ratio,
        TOKEN_STRIDE=token_stride,
        SCALE_DIM=scale_dim,
        KV_BLOCK_STRIDE=k_cache.stride(0),
        FP8_MAX=448.0,
        QUANT_MODE=quant_mode,
        FP4_ASM=_FP4_ASM_DEFAULT,
        CS_ALIGNED=(cos_sin_cache.stride(0) % 64) == 0,
        TP=tp,
        num_warps=1,
    )


@triton.jit
def _indexer_k_norm_rope_quant_store_kernel(
    k_pre_ptr,
    k_pre_stride,
    positions_ptr,
    rms_norm_weight_ptr,
    rms_norm_eps,
    cos_sin_cache_ptr,
    cos_sin_stride,
    k_cache_ptr,
    kv_slot_mapping_ptr,
    num_tokens,
    kv_cache_block_size,
    HEAD_SIZE: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    KV_BLOCK_STRIDE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    QUANT_MODE: tl.constexpr,
    FP4_ASM: tl.constexpr,
    CS_ALIGNED: tl.constexpr,
    TP: tl.constexpr,
):
    # ── Token block: [TP] rows, masked instead of early returns ───────
    pid = tl.program_id(0)
    tok = pid * TP + tl.arange(0, TP)
    tok_valid = tok < num_tokens
    slots = tl.load(kv_slot_mapping_ptr + tok, mask=tok_valid, other=-1)
    positions = tl.load(positions_ptr + tok, mask=tok_valid, other=0)
    active = tok_valid & (slots >= 0)
    if COMPRESS_RATIO != 1:
        # Only the last token of a group publishes that group's index key.
        active = active & (((positions + 1) % COMPRESS_RATIO) == 0)

    NUM_PAIRS: tl.constexpr = HEAD_SIZE // 2
    NOPE_PAIRS: tl.constexpr = (HEAD_SIZE - ROPE_HEAD_DIM) // 2
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2
    tl.static_assert(NOPE_PAIRS == HALF_ROPE)

    # ── cos/sin table: two 32-wide affine loads, hoisted above k_pre ──
    # A latent stands for the first token of its group, so group j takes
    # position j * compress_ratio. Loads are unmasked: every in-range
    # token's position must be covered by the table (caller contract —
    # rows beyond num_tokens read position 0 via the masked load above).
    # Masking these loads costs a select per element and measured ~15%
    # slower on C600-U; skipped tokens' lanes produce garbage that is
    # filtered by the active mask on every store.
    compressed_pos = (positions // COMPRESS_RATIO) * COMPRESS_RATIO
    off = compressed_pos * cos_sin_stride
    if CS_ALIGNED:
        off = tl.multiple_of(off, 64)
    block = tl.arange(0, HEAD_SIZE)
    c_raw = tl.load(
        cos_sin_cache_ptr + off[:, None] + tl.arange(0, HALF_ROPE)[None, :]
    ).to(tl.float32)
    s_raw = tl.load(
        cos_sin_cache_ptr
        + off[:, None]
        + (HALF_ROPE + tl.arange(0, HALF_ROPE))[None, :]
    ).to(tl.float32)

    # ── k_norm (fp32 throughout, bf16 roundtrip like the reference) ────
    rms_w = tl.load(rms_norm_weight_ptr + block).to(tl.float32)
    k = tl.load(
        k_pre_ptr + tok[:, None] * k_pre_stride + block[None, :],
        mask=active[:, None],
        other=0.0,
    ).to(tl.float32)
    variance = tl.sum(k * k, axis=1) / HEAD_SIZE
    k = (k * tl.rsqrt(variance + rms_norm_eps)[:, None] * rms_w[None, :]).to(
        tl.bfloat16
    )
    k = k.to(tl.float32)

    # ── Register-based GPT-J forward RoPE in fp32 ─────────────────────
    # Pairs [0, NOPE_PAIRS) pass through (cos = 1, sin = 0); the rest rotate
    # with the table row. The constants are joined with the loaded halves so
    # the whole transform stays gather- and select-free.
    even, odd = tl.split(tl.reshape(k, (TP, NUM_PAIRS, 2)))  # each [TP, 64]
    ones_v = tl.full((TP, HALF_ROPE), 1.0, tl.float32)
    zeros_v = tl.zeros((TP, HALF_ROPE), tl.float32)
    cos_all = tl.reshape(tl.permute(tl.join(ones_v, c_raw), (0, 2, 1)), (TP, NUM_PAIRS))
    sin_all = tl.reshape(
        tl.permute(tl.join(zeros_v, s_raw), (0, 2, 1)), (TP, NUM_PAIRS)
    )

    new_even = even * cos_all - odd * sin_all
    new_odd = odd * cos_all + even * sin_all

    # bf16 roundtrip for parity with the reference / Q-side kernel numerics.
    new_even = new_even.to(tl.bfloat16).to(tl.float32)
    new_odd = new_odd.to(tl.bfloat16).to(tl.float32)
    result = tl.interleave(new_even, new_odd)  # [TP, HEAD_SIZE] fp32

    # ── Paged cache pointers (segregated: values first, then scales) ──
    kv_block_idx = (slots // kv_cache_block_size).to(tl.int64)
    kv_pos_in_block = slots % kv_cache_block_size
    cache_block_ptr = k_cache_ptr + kv_block_idx * KV_BLOCK_STRIDE
    val_ptr = cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE
    scale_ptr = (
        cache_block_ptr
        + kv_cache_block_size * TOKEN_STRIDE
        + kv_pos_in_block * SCALE_DIM
    )

    if QUANT_MODE == 1:
        # ── MXFP4: 32-element blocks = 16 consecutive even/odd pairs ──
        N_QUANT_BLOCKS: tl.constexpr = HEAD_SIZE // 32
        HALF_BLOCK: tl.constexpr = 16
        even_2d = tl.reshape(new_even, (TP, N_QUANT_BLOCKS, HALF_BLOCK))
        odd_2d = tl.reshape(new_odd, (TP, N_QUANT_BLOCKS, HALF_BLOCK))

        amax = tl.maximum(
            tl.max(tl.abs(even_2d), axis=2),
            tl.max(tl.abs(odd_2d), axis=2),
        )
        amax = tl.maximum(amax, 6.0 * (2**-126))
        # ue8m0 block scale: 2^ceil(log2(amax / 6.0)), stored (exp + 127).
        log2_ratio = tl.ceil(tl.log2(amax * (1.0 / 6.0)))
        log2_ratio = tl.minimum(tl.maximum(log2_ratio, -127.0), 127.0)
        inv_scale = tl.exp2(-log2_ratio)
        ue8m0 = (log2_ratio + 127.0).to(tl.uint8)  # [TP, N_QUANT_BLOCKS]

        inv_scale_col = inv_scale[:, :, None]
        if FP4_ASM:
            packed = _fp32x2_to_fp4x2(
                even_2d * inv_scale_col, odd_2d * inv_scale_col
            )  # (TP, N_BLOCKS, HALF_BLOCK) uint8
        else:
            # Bit-identical software pack for assemblers without
            # cvt.rn.satfinite.e2m1x2 (MACA).
            packed = _fp32x2_to_fp4x2_sw(
                even_2d * inv_scale_col, odd_2d * inv_scale_col
            )
        packed_flat = tl.reshape(packed, (TP, TOKEN_STRIDE))

        tl.store(
            val_ptr[:, None] + tl.arange(0, TOKEN_STRIDE)[None, :],
            packed_flat,
            mask=active[:, None],
        )
        tl.store(
            scale_ptr[:, None] + tl.arange(0, SCALE_DIM)[None, :],
            ue8m0,
            mask=active[:, None],
        )
    elif QUANT_MODE == 0:
        # ── Per-token FP8 (single 128-wide block), one fp32 scale ────
        INV_FP8_MAX: tl.constexpr = 1.0 / FP8_MAX
        absmax = tl.maximum(tl.max(tl.abs(result), axis=1), 1e-4)
        exponent = tl.ceil(tl.log2(absmax * INV_FP8_MAX))
        inv_scale = tl.exp2(-exponent)
        # inv_scale is an exact power of two, so the product is exact and
        # |result * inv_scale| <= FP8_MAX by construction — no clamp needed.
        x_uint8 = (
            (result * inv_scale[:, None]).to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        )
        scale_bits = tl.exp2(exponent).to(tl.uint32, bitcast=True)
        scale_bytes = (
            (scale_bits[:, None] >> (8 * tl.arange(0, 4)[None, :])) & 0xFF
        ).to(tl.uint8)
        tl.store(
            val_ptr[:, None] + block[None, :],
            x_uint8,
            mask=active[:, None],
        )
        tl.store(
            scale_ptr[:, None] + tl.arange(0, SCALE_DIM)[None, :],
            scale_bytes,
            mask=active[:, None],
        )
    else:
        # ── Per-token symmetric INT8: one 128-wide block, one fp32 scale ─
        amax = tl.maximum(tl.max(tl.abs(result), axis=1), 1e-4)  # [TP]
        inv_scale = 127.0 / amax
        qf = result * inv_scale[:, None]
        # Round-to-nearest-even integer via the fp32 magic-number add; the
        # bitcast/subtract pair recovers the two's-complement value for
        # |qf| < 2**22, which always holds here.
        qi = (qf + 12582912.0).to(tl.uint32, bitcast=True) - 0x4B400000
        q = qi.to(tl.int8)
        codes = q.to(tl.uint8, bitcast=True)
        scale_bits = (amax / 127.0).to(tl.uint32, bitcast=True)
        scale_bytes = (
            (scale_bits[:, None] >> (8 * tl.arange(0, 4)[None, :])) & 0xFF
        ).to(tl.uint8)

        tl.store(
            val_ptr[:, None] + block[None, :],
            codes,
            mask=active[:, None],
        )
        tl.store(
            scale_ptr[:, None] + tl.arange(0, SCALE_DIM)[None, :],
            scale_bytes,
            mask=active[:, None],
        )
