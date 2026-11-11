# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.triton_utils import tl, triton


# -- fused_kv_compress_norm_rope_insert_sparse_attn_bf16 --
@triton.jit
def _fused_kv_compress_norm_rope_insert_sparse_attn_bf16(
    # ── state cache (compressor internal state) ──
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    # ── metadata ──
    token_to_req_indices_ptr,
    positions_ptr,
    slot_mapping_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    # ── RMSNorm ──
    rms_norm_weight_ptr,
    rms_norm_eps,
    # ── RoPE ──
    cos_sin_cache_ptr,
    cos_sin_stride,
    # ── KV cache output ──
    k_cache_ptr,
    kv_slot_mapping_ptr,
    kv_cache_block_size,
    # ── constexprs ──
    HEAD_SIZE: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    INT8_MAX: tl.constexpr,  # 127.0
    QUANT_BLOCK: tl.constexpr,  # 64 for DeepseekV4
    TOKEN_STRIDE: tl.constexpr,  # Physical BF16 elements per token
    SCALE_DIM: tl.constexpr,  # Unused; kept for the shared launcher signature
    KV_BLOCK_STRIDE: tl.constexpr,
):
    """Fused compress → RMSNorm → RoPE → bf16 store.

    One program per token; early-exits for non-boundary positions.

    Cache block layout (``block_size`` tokens):
      [0, bs*HEAD_SIZE): bf16 token data (HEAD_SIZE elements)
    """
    token_idx = tl.program_id(0)

    slot_id = tl.load(slot_mapping_ptr + token_idx)
    if slot_id < 0:
        return

    position = tl.load(positions_ptr + token_idx)
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    # ── Gather state cache entries ────────────────────────────────────
    start = position - (1 + OVERLAP) * COMPRESS_RATIO + 1
    tokens = tl.arange(0, (1 + OVERLAP) * COMPRESS_RATIO)
    pos = start + tokens
    mask_pos = pos >= 0

    block_indices = pos // block_size
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + block_indices,
        mask=mask_pos,
        other=0,
    )
    block_offsets = pos % block_size
    head_offset = (tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE
    block_numbers_i64 = block_numbers.to(tl.int64)

    # Precomputed row base shared by score and kv loads
    row_base = (
        state_cache_ptr
        + block_numbers_i64 * state_cache_stride0
        + block_offsets * state_cache_stride1
        + head_offset
    )

    combined_mask = mask_pos[:, None] & mask[None, :]

    # ── Softmax + weighted sum ───────────────────────────────────────
    score = tl.load(
        row_base[:, None] + STATE_WIDTH + block[None, :],
        mask=combined_mask,
        other=float("-inf"),
    )
    score = tl.softmax(score, dim=0)

    kv = tl.load(
        row_base[:, None] + block[None, :],
        mask=combined_mask,
        other=0.0,
    )

    compressed_kv = tl.sum(kv * score, axis=0)  # [TRITON_BLOCK_SIZE] fp32

    # ── RMSNorm (fp32 throughout) ──────────────────────────────────────
    rms_w = tl.load(rms_norm_weight_ptr + block, mask=mask, other=0.0)
    variance = tl.sum(compressed_kv * compressed_kv, axis=0) / HEAD_SIZE
    rrms = tl.rsqrt(variance + rms_norm_eps)
    normed = compressed_kv * rrms * rms_w

    # ── KV cache pointers ───────────────────────────────────────────
    kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx)
    if kv_slot_idx < 0:
        return
    kv_block_idx = kv_slot_idx // kv_cache_block_size
    kv_pos_in_block = kv_slot_idx % kv_cache_block_size

    # Both strides are taken from the concrete cache view and are expressed in
    # BF16 elements. In particular, TOKEN_STRIDE must not be the upstream
    # 576-byte FP8 packed-row constant.
    tl.static_assert(
        TOKEN_STRIDE >= HEAD_SIZE,
        "BF16 KV token stride must cover the full attention head",
    )
    cache_block_ptr = k_cache_ptr + kv_block_idx.to(tl.int64) * KV_BLOCK_STRIDE
    # ── MC3-13058 ────────────────────────────────────────────────
    # The inherited TOKEN_STRIDE=576 targets the packed-FP8 layout:
    # 448 NoPE + 64 RoPE * 2. The MetaX BF16 cache stores HEAD_SIZE
    # elements per row, so using TOKEN_STRIDE would misalign later rows
    # and corrupt the decode KV cache.
    TOKEN_STRIDE = HEAD_SIZE
    bf16_ptr = cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE

    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM  # 448
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2  # 32

    # ── Register-based GPT-J forward RoPE in fp32 ─────────────────────
    # Apply RoPE to even/odd pairs, then restore the dense head order.
    NUM_PAIRS: tl.constexpr = TRITON_BLOCK_SIZE // 2
    NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2

    normed_2d = tl.reshape(normed, (NUM_PAIRS, 2))
    even, odd = tl.split(normed_2d)  # each [NUM_PAIRS] fp32

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
    result = tl.interleave(new_even, new_odd)  # [TRITON_BLOCK_SIZE] fp32

    # Store entire result as bf16 (both nope and rope portions)
    tl.store(bf16_ptr + block, result.to(tl.bfloat16), mask=mask)
