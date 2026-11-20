# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch

from vllm_metax.platform import MacaPlatform

pytestmark = pytest.mark.requires_maca


def _rope(x, cos, sin):
    body, tail = x.float().split([448, 64], dim=-1)
    even, odd = tail[..., ::2], tail[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1)
    return torch.cat((body, rotated.flatten(-2)), dim=-1)


@pytest.mark.parametrize("num_tokens", [4, 17])
@pytest.mark.parametrize("heads", [8, 17])
@pytest.mark.parametrize("apply_q_norm", [True, False])
def test_full_cache_bf16_matches_reference(device, num_tokens, heads, apply_q_norm):
    MacaPlatform.import_kernels()
    rng = torch.Generator(device=device).manual_seed(5)
    q = torch.randn(
        num_tokens, heads, 512, dtype=torch.bfloat16, device=device, generator=rng
    )
    kv = torch.randn(num_tokens, 512, dtype=q.dtype, device=device, generator=rng)
    expected_q = q.float()
    eps = 1e-6
    if apply_q_norm:
        expected_q = expected_q * torch.rsqrt(
            expected_q.square().mean(-1, keepdim=True) + eps
        )
    angles = torch.randn(
        num_tokens + 3, 32, dtype=torch.float32, device=device, generator=rng
    )
    cos, sin = angles.cos(), angles.sin()
    trig = torch.cat((cos, sin), dim=-1)
    positions = torch.arange(3, 3 + num_tokens, device=device, dtype=torch.int64)
    expected_q = _rope(expected_q, cos[positions, None], sin[positions, None]).to(
        q.dtype
    )
    expected_kv = _rope(kv, cos[positions], sin[positions]).to(kv.dtype)
    slots = torch.arange(num_tokens, device=device, dtype=torch.int64).flip(0) + 2
    slots[1] = -1
    block_size = 16
    cache = torch.full((2, block_size, 512), -7, device=device, dtype=q.dtype)
    expected_cache = torch.full_like(cache, -7)
    valid = slots >= 0
    expected_cache.flatten(0, 1)[slots[valid]] = expected_kv[valid]

    torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_bf16_insert(
        q, kv, cache, slots, positions, trig, eps, block_size, apply_q_norm
    )

    torch.testing.assert_close(q, expected_q, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(cache, expected_cache, atol=0, rtol=0)
