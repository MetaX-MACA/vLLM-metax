# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""MACA MXFP4 Q quantization for the DeepSeek-V4/V4.1 indexer."""

import torch

from vllm.triton_utils import tl, triton

MXFP4_BLOCK_SIZE = 32


@triton.jit
def _fp32_to_e2m1_nibble(x):
    # E2M1 magnitudes: 0, .5, 1, 1.5, 2, 3, 4, 6.
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
    return _fp32_to_e2m1_nibble(x_lo) | (_fp32_to_e2m1_nibble(x_hi) << 4)


@triton.jit
def _quantize_mxfp4_pair(x_lo, x_hi):
    amax = tl.maximum(tl.max(tl.abs(x_lo)), tl.max(tl.abs(x_hi)))
    amax = tl.maximum(amax, 6.0 * (2**-126))
    log2_ratio = tl.ceil(tl.log2(amax * (1.0 / 6.0)))
    log2_ratio = tl.minimum(tl.maximum(log2_ratio, -127.0), 127.0)
    inv_scale = tl.exp2(-log2_ratio)
    ue8m0 = (log2_ratio + 127.0).to(tl.uint8)
    packed = _fp32x2_to_fp4x2_sw(x_lo * inv_scale, x_hi * inv_scale)
    return packed, ue8m0


@triton.jit
def _fused_indexer_q_rope_mxfp4_quant_kernel(
    positions_ptr,
    index_q_ptr,
    index_q_stride0,
    index_q_stride1,
    cos_sin_ptr,
    cos_sin_stride,
    index_q_packed_ptr,
    index_q_packed_stride0,
    index_q_packed_stride1,
    index_q_scale_ptr,
    index_q_scale_stride0,
    index_q_scale_stride1,
    index_weights_ptr,
    index_weights_stride,
    index_weights_softmax_scale,
    index_weights_head_scale,
    index_weights_out_ptr,
    index_weights_out_stride,
    HEAD_DIM: tl.constexpr,
    HALF_ROT_DIM: tl.constexpr,
    MXFP4_BLOCK: tl.constexpr,
):
    rot_dim: tl.constexpr = 2 * HALF_ROT_DIM
    nope_dim: tl.constexpr = HEAD_DIM - rot_dim
    num_nope_blocks: tl.constexpr = nope_dim // MXFP4_BLOCK
    num_rope_blocks: tl.constexpr = rot_dim // MXFP4_BLOCK
    half_block: tl.constexpr = MXFP4_BLOCK // 2
    tl.static_assert(nope_dim >= 0)
    tl.static_assert(nope_dim % MXFP4_BLOCK == 0)
    tl.static_assert(rot_dim % MXFP4_BLOCK == 0)

    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    position = tl.load(positions_ptr + token_idx)

    input_base = index_q_ptr + token_idx * index_q_stride0 + head_idx * index_q_stride1
    packed_base = (
        index_q_packed_ptr
        + token_idx * index_q_packed_stride0
        + head_idx * index_q_packed_stride1
    )
    scale_base = (
        index_q_scale_ptr
        + token_idx * index_q_scale_stride0
        + head_idx * index_q_scale_stride1
    )
    half_offset = tl.arange(0, half_block)

    # Leading NoPE dimensions pass through unchanged, quantized per 32 values.
    for block_idx in tl.static_range(num_nope_blocks):
        base = block_idx * MXFP4_BLOCK
        x_lo = tl.load(input_base + base + half_offset * 2).to(tl.float32)
        x_hi = tl.load(input_base + base + half_offset * 2 + 1).to(tl.float32)
        packed, ue8m0 = _quantize_mxfp4_pair(x_lo, x_hi)
        tl.store(packed_base + base // 2 + half_offset, packed)
        tl.store(scale_base + block_idx, ue8m0)

    # GPT-J interleaved RoPE applies only to the trailing ROPE dimensions.
    rot_input_base = input_base + nope_dim
    cos_sin_row = cos_sin_ptr + position * cos_sin_stride
    for block_idx in tl.static_range(num_rope_blocks):
        pair_offset = block_idx * half_block + half_offset
        cos = tl.load(cos_sin_row + pair_offset).to(tl.float32)
        sin = tl.load(cos_sin_row + pair_offset + HALF_ROT_DIM).to(tl.float32)
        x_even = tl.load(rot_input_base + pair_offset * 2).to(tl.float32)
        x_odd = tl.load(rot_input_base + pair_offset * 2 + 1).to(tl.float32)
        r_even = x_even * cos - x_odd * sin
        r_odd = x_odd * cos + x_even * sin
        r_even = r_even.to(tl.bfloat16).to(tl.float32)
        r_odd = r_odd.to(tl.bfloat16).to(tl.float32)
        packed, ue8m0 = _quantize_mxfp4_pair(r_even, r_odd)
        packed_offset = (nope_dim + block_idx * MXFP4_BLOCK) // 2
        tl.store(packed_base + packed_offset + half_offset, packed)
        tl.store(scale_base + num_nope_blocks + block_idx, ue8m0)

    # MXFP4 keeps its per-32 UE8M0 scales separate. Only softmax/head scales
    # are folded into the per-head weights.
    weight = tl.load(
        index_weights_ptr + token_idx * index_weights_stride + head_idx
    ).to(tl.float32)
    weight *= index_weights_softmax_scale
    weight *= index_weights_head_scale
    tl.store(
        index_weights_out_ptr + token_idx * index_weights_out_stride + head_idx,
        weight,
    )


def fused_indexer_q_rope_mxfp4_quant(
    positions: torch.Tensor,
    index_q: torch.Tensor,
    index_q_cos_sin_cache: torch.Tensor,
    index_weights: torch.Tensor,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
    weights_out_dtype: torch.dtype = torch.float32,
) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Fuse GPT-J RoPE and MXFP4 quantization without MACA-unsupported PTX."""
    assert positions.ndim == 1
    assert index_q.ndim == 3
    assert index_q_cos_sin_cache.ndim == 2
    assert index_q.stride(-1) == 1
    assert index_q_cos_sin_cache.stride(-1) == 1
    assert index_q.shape[0] == positions.shape[0]
    assert index_weights.shape == (positions.shape[0], index_q.shape[1])

    num_tokens, num_heads, head_dim = index_q.shape
    rope_dim = index_q_cos_sin_cache.shape[-1]
    assert head_dim % MXFP4_BLOCK_SIZE == 0
    assert rope_dim == 64 and head_dim >= rope_dim
    assert (head_dim - rope_dim) % MXFP4_BLOCK_SIZE == 0

    packed = torch.empty(
        (num_tokens, num_heads, head_dim // 2),
        dtype=torch.uint8,
        device=index_q.device,
    )
    scale = torch.empty(
        (num_tokens, num_heads, head_dim // MXFP4_BLOCK_SIZE),
        dtype=torch.uint8,
        device=index_q.device,
    )
    weights_out = torch.empty_like(index_weights, dtype=weights_out_dtype)

    if num_tokens == 0:
        return (packed, scale.view(torch.int32).squeeze(-1)), weights_out

    _fused_indexer_q_rope_mxfp4_quant_kernel[(num_tokens, num_heads)](
        positions,
        index_q,
        index_q.stride(0),
        index_q.stride(1),
        index_q_cos_sin_cache,
        index_q_cos_sin_cache.stride(0),
        packed,
        packed.stride(0),
        packed.stride(1),
        scale,
        scale.stride(0),
        scale.stride(1),
        index_weights,
        index_weights.stride(0),
        index_weights_softmax_scale,
        index_weights_head_scale,
        weights_out,
        weights_out.stride(0),
        HEAD_DIM=head_dim,
        HALF_ROT_DIM=rope_dim // 2,
        MXFP4_BLOCK=MXFP4_BLOCK_SIZE,
        num_warps=1,
    )
    return (packed, scale.view(torch.int32).squeeze(-1)), weights_out
