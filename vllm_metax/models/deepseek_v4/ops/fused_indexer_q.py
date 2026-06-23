# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch
from vllm.triton_utils import tl, triton

@triton.jit
def _round_to_nearest(x):
    # round-half-away-from-zero
    return tl.where(x >= 0.0, tl.floor(x + 0.5), tl.ceil(x - 0.5))


@triton.jit
def _quantize_int8(x, scale):
    q = tl.div_rn(x, scale)
    q = _round_to_nearest(q)
    q = tl.maximum(tl.minimum(q, 127.0), -127.0)
    return q.to(tl.int8)


@triton.jit
def _get_cos_sin(
    cos_sin_cache_ptr,
    cos_sin_cache_stride,
    pos,
    HALF_ROT_DIM: tl.constexpr,
):
    block = tl.arange(0, HALF_ROT_DIM)
    cos = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + block)
    cos = cos.to(tl.float32)
    sin = tl.load(cos_sin_cache_ptr + pos * cos_sin_cache_stride + block + HALF_ROT_DIM)
    sin = sin.to(tl.float32)
    return cos, sin


@triton.jit
def _fused_indexer_q_rope_int8_quant_kernel(
    pos_ptr,
    # Index Q RoPE
    index_q_ptr,
    index_q_stride0,
    index_q_stride1,
    index_q_cos_sin_ptr,
    index_q_cos_sin_stride,
    INDEX_Q_HALF_ROT_DIM: tl.constexpr,
    # Index Q Quantize
    index_q_int8_ptr,
    index_q_int8_stride0,
    index_q_int8_stride1,
    INDEX_Q_HEAD_DIM: tl.constexpr,
    # Index weights
    index_weights_ptr,
    index_weights_stride,
    index_weights_softmax_scale,
    index_weights_head_scale,
    index_weights_out_ptr,
    index_weights_out_stride,
):
    INDEX_Q_ROT_DIM: tl.constexpr = 2 * INDEX_Q_HALF_ROT_DIM
    INDEX_Q_NOPE_DIM: tl.constexpr = INDEX_Q_HEAD_DIM - INDEX_Q_ROT_DIM
    tl.static_assert(INDEX_Q_NOPE_DIM >= 0)

    tok_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    pos = tl.load(pos_ptr + tok_idx)
    cos, sin = _get_cos_sin(
        index_q_cos_sin_ptr,
        index_q_cos_sin_stride,
        pos,
        INDEX_Q_HALF_ROT_DIM,
    )

    half_offset = tl.arange(0, INDEX_Q_HALF_ROT_DIM)
    base_ptr = index_q_ptr + tok_idx * index_q_stride0 + head_idx * index_q_stride1

    # GPT-J interleaved RoPE on dims [NOPE_DIM, HEAD_DIM)
    rot_base = base_ptr + INDEX_Q_NOPE_DIM
    x_even = tl.load(rot_base + half_offset * 2).to(tl.float32)
    x_odd = tl.load(rot_base + half_offset * 2 + 1).to(tl.float32)

    r_even = x_even * cos - x_odd * sin
    r_odd = x_odd * cos + x_even * sin

    # Keep same numeric convention as original FP8 path:
    # fp32 -> bf16 -> fp32 before absmax / quant.
    r_even = r_even.to(tl.bfloat16).to(tl.float32)
    r_odd = r_odd.to(tl.bfloat16).to(tl.float32)

    amax = tl.maximum(tl.max(tl.abs(r_even)), tl.max(tl.abs(r_odd)))

    if INDEX_Q_NOPE_DIM > 0:
        nope_offset = tl.arange(0, INDEX_Q_NOPE_DIM)
        x_nope = tl.load(base_ptr + nope_offset).to(tl.float32)
        amax = tl.maximum(amax, tl.max(tl.abs(x_nope)))

    # INT8 symmetric scale.
    index_q_scale = tl.where(amax > 0.0, amax / 127.0, 1.0)

    int8_base_ptr = (
        index_q_int8_ptr
        + tok_idx * index_q_int8_stride0
        + head_idx * index_q_int8_stride1
    )

    if INDEX_Q_NOPE_DIM > 0:
        tl.store(
            int8_base_ptr + nope_offset,
            _quantize_int8(x_nope, index_q_scale),
        )

    int8_rot_base = int8_base_ptr + INDEX_Q_NOPE_DIM

    tl.store(
        int8_rot_base + half_offset * 2,
        _quantize_int8(r_even, index_q_scale),
    )

    tl.store(
        int8_rot_base + half_offset * 2 + 1,
        _quantize_int8(r_odd, index_q_scale),
    )

    # INT8 weight-fold contract:
    #   q_int8 approximately represents q / index_q_scale
    #   q_original approximately q_int8 * index_q_scale
    #
    # Since the Q scale is not returned as a separate tensor, fold it into
    # index_weights_out, same as the old FP8 path.
    index_weights = tl.load(
        index_weights_ptr + tok_idx * index_weights_stride + head_idx
    ).to(tl.float32)

    index_weights *= index_q_scale
    index_weights *= index_weights_softmax_scale
    index_weights *= index_weights_head_scale

    tl.store(
        index_weights_out_ptr + tok_idx * index_weights_out_stride + head_idx,
        index_weights,
    )


def fused_indexer_q_rope_int8_quant(
    positions: torch.Tensor,
    index_q: torch.Tensor,
    index_q_cos_sin_cache: torch.Tensor,
    # Index weights
    index_weights: torch.Tensor,
    index_weights_softmax_scale: float,
    index_weights_head_scale: float,
) -> tuple[
    torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    torch.Tensor,
]:
    """Fused RoPE + quantize Q for the sparse indexer.

    Weight-fold semantics:

    INT8:
        q_int8     : (T, H, HEAD_DIM) int8, per-token-per-head scalar scale
                     is NOT stored — folded into weights below.
        weights_out = weights * q_scale * softmax_scale * head_scale

        Quantization:
            q_scale = max(abs(q_rope)) / 127
            q_int8  = clamp(round(q_rope / q_scale), -127, 127)

        Dequantization contract:
            q_rope ≈ q_int8.float() * q_scale

    """
    assert positions.ndim == 1
    assert index_q.ndim == 3
    assert index_q_cos_sin_cache.ndim == 2

    num_tokens = positions.shape[0]
    num_index_q_heads = index_q.shape[1]
    index_q_head_dim = index_q.shape[2]

    index_weights_out = torch.empty_like(index_weights, dtype=torch.float32)

    index_q_int8 = torch.empty_like(index_q, dtype=torch.int8)

    _fused_indexer_q_rope_int8_quant_kernel[(num_tokens, num_index_q_heads)](
        positions,
        index_q,
        index_q.stride(0),
        index_q.stride(1),
        index_q_cos_sin_cache,
        index_q_cos_sin_cache.stride(0),
        index_q_cos_sin_cache.shape[-1] // 2,
        index_q_int8,
        index_q_int8.stride(0),
        index_q_int8.stride(1),
        index_q_head_dim,
        index_weights,
        index_weights.stride(0),
        index_weights_softmax_scale,
        index_weights_head_scale,
        index_weights_out,
        index_weights_out.stride(0),
        num_warps=1,
    )

    return index_q_int8, index_weights_out
