# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch

from vllm_metax.models.minimax_m3.metax.attention_ops import (
    fused_minimax_m3_qknorm_rope_kv_insert,
)

pytestmark = pytest.mark.requires_maca


def _norm_rope(x, weight, cos, sin, eps):
    val = x.float()
    val = val * torch.rsqrt(val.square().mean(-1, keepdim=True) + eps)
    val = val * (1 + weight.float())
    first, second, tail = val.split(
        (cos.size(-1), cos.size(-1), val.size(-1) - 2 * cos.size(-1)), dim=-1
    )
    return torch.cat(
        (first * cos - second * sin, second * cos + first * sin, tail), dim=-1
    ).to(x.dtype)


@pytest.mark.parametrize("tokens", [1, 7, 64, 513])
@pytest.mark.parametrize("heads,kv_heads", [(8, 2), (16, 4), (64, 4)])
@torch.inference_mode()
def test_dense_norm_rope(device, tokens, heads, kv_heads):
    rng = torch.Generator(device=device).manual_seed(0)
    head, rotary, eps = 128, 64, 1e-6
    dtype = torch.bfloat16
    widths = [heads * head, kv_heads * head, kv_heads * head]
    qkv = torch.randn(tokens, sum(widths), dtype=dtype, device=device, generator=rng)
    q_w = torch.randn(head, dtype=dtype, device=device, generator=rng) / 10
    k_w = torch.randn_like(q_w) / 10
    angles = torch.randn(tokens + 9, rotary // 2, device=device, generator=rng)
    cos, sin = angles.cos().to(dtype), angles.sin().to(dtype)
    rope = torch.cat((cos, sin), dim=-1)
    positions = torch.randint(tokens + 9, (tokens,), device=device, generator=rng)
    q, k, v = qkv.split(widths, dim=-1)
    cos_rows, sin_rows = cos[positions, None].float(), sin[positions, None].float()
    expected_q = _norm_rope(q.unflatten(-1, (-1, head)), q_w, cos_rows, sin_rows, eps)
    expected_k = _norm_rope(k.unflatten(-1, (-1, head)), k_w, cos_rows, sin_rows, eps)
    expected_v = v.clone()

    fused_minimax_m3_qknorm_rope_kv_insert(
        qkv, q_w, k_w, rope, positions, heads, kv_heads, rotary, eps
    )

    torch.testing.assert_close(q, expected_q.flatten(1), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k, expected_k.flatten(1), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(v, expected_v, atol=0, rtol=0)
