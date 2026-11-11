"""Source-mode plain-row BF16 KV cache writer for xcore1002.

The vendor fused op ``_C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert``
ships only in precompiled wheels. This module reproduces it with plain torch:
Q-side norm+RoPE+head padding (identical math to the int8 source fallback in
``kv_cache_int8.py``) and KV-side RoPE + paged scatter into a bf16 row cache
``[num_blocks, block_size, head_dim]``.
"""

from __future__ import annotations

import torch

NOPE_DIM = 448
ROPE_DIM = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM  # 512


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    even, odd = x[..., 0::2], x[..., 1::2]
    rotated = torch.stack(
        (even * cos - odd * sin, even * sin + odd * cos), dim=-1
    ).flatten(-2)
    return rotated


def qnorm_rope_kv_bf16_insert(
    q: torch.Tensor,
    kv: torch.Tensor,
    k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    padded_heads: int,
    eps: float,
    block_size: int,
    apply_q_norm: bool,
) -> torch.Tensor:
    """Q(norm+RoPE+pad) + KV(RoPE+bf16 paged insert). Returns padded q.

    k_cache: ``[num_blocks, block_size, HEAD_DIM]`` bf16 rows.
    """
    assert q.ndim == 3 and q.shape[-1] == HEAD_DIM, (
        f"BF16 Q must be [tokens, heads, {HEAD_DIM}], got {tuple(q.shape)}"
    )
    assert k_cache.dtype == torch.bfloat16 and k_cache.shape[1:] == (
        block_size,
        HEAD_DIM,
    ), f"bf16 row cache expected, got {k_cache.dtype} {tuple(k_cache.shape)}"

    num_tokens = q.shape[0]
    q_out = torch.zeros(
        (num_tokens, padded_heads, HEAD_DIM), dtype=q.dtype, device=q.device
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
        nope = q_float[..., :NOPE_DIM]
        rotary = q_float[..., NOPE_DIM:]
        rotated = _rope(rotary, cos, sin)
        q_out[:, : q.shape[1]].copy_(
            torch.cat((nope, rotated), dim=-1).to(q.dtype)
        )

    if num_tokens:
        kv_float = kv.float()
        cos_sin = cos_sin_cache[positions].float()
        cos, sin = cos_sin.chunk(2, dim=-1)
        nope = kv_float[..., :NOPE_DIM]
        rotary = kv_float[..., NOPE_DIM:]
        rotated = _rope(rotary, cos, sin)
        kv_out = torch.cat((nope, rotated), dim=-1).to(torch.bfloat16)
        valid = slot_mapping >= 0
        slots = slot_mapping[valid]
        bs = k_cache.shape[1]
        k_cache[slots // bs, slots % bs] = kv_out[valid]

    return q_out
