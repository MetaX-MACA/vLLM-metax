# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch

from vllm_metax.platform import MacaPlatform


def _norm_and_rotate(x, weight, head_dim, eps, cos, sin, is_neox):
    heads = x.unflatten(-1, (-1, head_dim)).float()
    normalized = heads * torch.rsqrt(heads.square().mean(-1, keepdim=True) + eps)
    normalized = (normalized * weight.float()).to(x.dtype).float()
    rotated, tail = normalized.split(
        (2 * cos.shape[-1], head_dim - 2 * cos.shape[-1]), dim=-1
    )
    if is_neox:
        left, right = rotated.chunk(2, dim=-1)
        rotated = torch.cat(
            (left * cos - right * sin, right * cos + left * sin), dim=-1
        )
    else:
        even, odd = rotated[..., ::2], rotated[..., 1::2]
        rotated = torch.stack(
            (even * cos - odd * sin, odd * cos + even * sin), dim=-1
        ).flatten(-2)
    return torch.cat((rotated, tail), dim=-1).flatten(-2).to(x.dtype)


# Adapted from vLLM v0.33.0 tests/kernels/core/test_fused_qk_norm_rope.py.
@pytest.mark.requires_maca
@pytest.mark.parametrize("is_neox", [False, True])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
@pytest.mark.parametrize("rotary_dim", [32, 64, 128])
def test_native_fused_qk_norm_rope_updates_qk_only(device, is_neox, eps, rotary_dim):
    MacaPlatform.import_kernels()
    torch.manual_seed(13)
    dtype = torch.float16
    query_heads, kv_heads, head_dim, tokens = 16, 4, 128, 7
    widths = (query_heads * head_dim, kv_heads * head_dim, kv_heads * head_dim)
    qkv = torch.randn(tokens, sum(widths), device=device, dtype=dtype)
    qkv[0].zero_()
    q, k, v = qkv.split(widths, dim=1)
    q_weight = torch.randn(head_dim, device=device, dtype=dtype) / 8 + 1
    k_weight = torch.randn_like(q_weight) / 8 + 1
    positions = torch.tensor([0, 19, 1, 23, 2, 7, 19], device=device)
    inverse_freq = 10000 ** (
        -torch.arange(0, rotary_dim, 2, device=device).float() / rotary_dim
    )
    angles = torch.arange(32, device=device)[:, None] * inverse_freq
    cache = torch.cat((angles.cos(), angles.sin()), dim=-1).to(dtype)
    cos, sin = cache[positions, None].float().chunk(2, dim=-1)
    expected_q = _norm_and_rotate(q, q_weight, head_dim, eps, cos, sin, is_neox)
    expected_k = _norm_and_rotate(k, k_weight, head_dim, eps, cos, sin, is_neox)
    expected_v = v.clone()

    torch.ops._C.fused_qk_norm_rope(
        qkv,
        query_heads,
        kv_heads,
        kv_heads,
        head_dim,
        eps,
        q_weight,
        k_weight,
        cache,
        is_neox,
        positions,
    )

    atol = 2e-3
    torch.testing.assert_close(q, expected_q, atol=atol, rtol=atol)
    torch.testing.assert_close(k, expected_k, atol=atol, rtol=atol)
    torch.testing.assert_close(v, expected_v, atol=0, rtol=0)
