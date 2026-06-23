# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch
from vllm.triton_utils import tl, triton


# -------- inv_rope ---------
@triton.jit
def _inv_rope_kernel(
    o_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    out_ptr,
    num_tokens,
    heads_per_group: tl.constexpr,
    o_stride_token,
    o_stride_head,
    cache_stride_pos,
    out_stride_group,
    out_stride_token,
    out_stride_d,
    CHUNKS_PER_HEAD: tl.constexpr,
    QUANT_GROUP_SIZE: tl.constexpr,
    ROPE_START: tl.constexpr,
    HALF_ROPE: tl.constexpr,
):
    # int64: stride multiply overflows int32 past num_tokens=32768.
    pid_token = tl.program_id(0).to(tl.int64)
    pid_gh = tl.program_id(1).to(tl.int64)

    g = pid_gh // heads_per_group
    head_in_group = pid_gh % heads_per_group
    global_head = pid_gh

    HEAD_DIM: tl.constexpr = CHUNKS_PER_HEAD * QUANT_GROUP_SIZE
    offsets = tl.arange(0, HEAD_DIM)

    input_base = o_ptr + pid_token * o_stride_token + global_head * o_stride_head

    x = tl.load(input_base + offsets).to(tl.float32)

    # RoPE starts at absolute offset:
    # default: nope_dim=448, rope_dim=64, head_dim=512
    # quant_group_size=128, chunks_per_head=4, rope_start=64
    # rope_abs_start = 3 * 128 + 64 = 448
    rope_abs_start: tl.constexpr = (CHUNKS_PER_HEAD - 1) * QUANT_GROUP_SIZE + ROPE_START

    pos = tl.load(positions_ptr + pid_token)
    cache_base = cos_sin_cache_ptr + pos * cache_stride_pos

    is_rope = offsets >= rope_abs_start
    rope_local = offsets - rope_abs_start

    x_partner = tl.load(
        input_base + (offsets ^ 1),
        mask=is_rope,
        other=0.0,
    ).to(tl.float32)

    cs_idx = tl.maximum(rope_local >> 1, 0)

    cos_v = tl.load(cache_base + cs_idx, mask=is_rope, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)

    # inverse RoPE:
    # even: x_even * cos + x_odd  * sin
    # odd : x_odd  * cos - x_even * sin
    x_add = x * cos_v + x_partner * sin_v
    x_sub = x * cos_v - x_partner * sin_v

    is_even = (rope_local & 1) == 0
    rotated = tl.where(is_even, x_add, x_sub)
    y = tl.where(is_rope, rotated, x)

    out_base = (
        out_ptr
        + g * out_stride_group
        + pid_token * out_stride_token
        + head_in_group * HEAD_DIM
    )

    tl.store(out_base + offsets * out_stride_d, y)


def inv_rope(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int = 448,
    rope_dim: int = 64,
    quant_group_size: int = 128,
) -> torch.Tensor:
    """Fused inverse RoPE without quantization.

    Args:
        o:
            Attention output, shape [num_tokens, num_heads, head_dim].
            dtype can be bf16/fp16/fp32.
        positions:
            Token positions, shape [num_tokens], int64/int32.
        cos_sin_cache:
            Precomputed cos||sin cache, shape [max_pos, rope_dim], fp32.
        n_groups:
            Number of output groups.
        heads_per_group:
            Number of heads per group.
        nope_dim:
            Non-RoPE dimensions per head.
        rope_dim:
            RoPE dimensions per head.
        quant_group_size:
            Kept only for compatibility with original layout assumptions.
            It no longer means quantization group size here.

    Returns:
        out:
            Shape [num_tokens, n_groups, heads_per_group * head_dim].
            dtype is same as input o.
    """
    num_tokens, num_heads, head_dim = o.shape

    assert num_heads == n_groups * heads_per_group
    assert head_dim == nope_dim + rope_dim
    assert head_dim % quant_group_size == 0
    assert nope_dim % quant_group_size == (quant_group_size - rope_dim)
    assert rope_dim % 2 == 0
    assert cos_sin_cache.shape[-1] == rope_dim
    assert cos_sin_cache.dtype == torch.float32

    d = heads_per_group * head_dim
    chunks_per_head = head_dim // quant_group_size

    out_buf = torch.empty(
        (n_groups, num_tokens, d),
        dtype=o.dtype,
        device=o.device,
    )

    grid = (num_tokens, n_groups * heads_per_group)

    _inv_rope_kernel[grid](
        o,
        positions,
        cos_sin_cache,
        out_buf,
        num_tokens,
        heads_per_group,
        o.stride(0),
        o.stride(1),
        cos_sin_cache.stride(0),
        out_buf.stride(0),
        out_buf.stride(1),
        out_buf.stride(2),
        CHUNKS_PER_HEAD=chunks_per_head,
        QUANT_GROUP_SIZE=quant_group_size,
        ROPE_START=nope_dim % quant_group_size,
        HALF_ROPE=rope_dim // 2,
        num_warps=1,
        num_stages=1,
    )

    return out_buf.transpose(0, 1)


# ---------------------------