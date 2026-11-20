# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch

from vllm_metax.platform import MacaPlatform

pytestmark = pytest.mark.requires_maca


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("is_neox_style", [False, True])
@pytest.mark.parametrize("rope_dim", [64, 128])
def test_concat_and_cache_mla_rope_fused(device, dtype, is_neox_style, rope_dim):
    MacaPlatform.import_kernels()
    rng = torch.Generator(device=device).manual_seed(42)

    def values(*shape):
        return torch.randn(*shape, dtype=dtype, device=device, generator=rng)

    positions = torch.tensor([8, 3, 1, 13, 2], dtype=torch.int64, device=device)
    q = values(5, 6, rope_dim)
    k = values(5, 2 * rope_dim)[:, :rope_dim]
    kv = values(5, 512)
    angles = torch.randn(
        16, rope_dim // 2, dtype=torch.float32, device=device, generator=rng
    )
    cos, sin = angles.cos().to(dtype), angles.sin().to(dtype)
    cache = torch.full((4, 16, 512 + rope_dim), -9, dtype=dtype, device=device)
    slots = torch.tensor([18, 63, -1, 2, 33], dtype=torch.int64, device=device)

    def reference(x, c, s):
        x = x.float()
        if is_neox_style:
            first, second = x.chunk(2, dim=-1)
        else:
            first, second = x[..., ::2], x[..., 1::2]
        real = first * c - second * s
        imag = first * s + second * c
        if is_neox_style:
            return torch.cat((real, imag), dim=-1).to(dtype)
        return torch.stack((real, imag), dim=-1).flatten(-2).to(dtype)

    expected_q = reference(q, cos[positions, None], sin[positions, None])
    expected_k = reference(k, cos[positions], sin[positions])
    valid = slots >= 0
    expected_cache = torch.full_like(cache, -9)
    expected_cache.flatten(0, 1)[slots[valid]] = torch.cat(
        (kv[valid], expected_k[valid]), dim=-1
    )
    original_q = q.clone()
    original_k = k.clone()

    torch.ops._C_cache_ops.concat_and_cache_mla_rope_fused(
        positions,
        q,
        k,
        kv,
        torch.cat((cos, sin), dim=-1),
        is_neox_style,
        slots,
        cache,
        "auto",
        torch.ones(1, device=device),
    )

    torch.testing.assert_close(q[valid], expected_q[valid], atol=1e-2, rtol=1.6e-2)
    torch.testing.assert_close(k[valid], expected_k[valid], atol=1e-2, rtol=1.6e-2)
    torch.testing.assert_close(q[~valid], original_q[~valid], atol=0, rtol=0)
    torch.testing.assert_close(k[~valid], original_k[~valid], atol=0, rtol=0)
    torch.testing.assert_close(cache, expected_cache, atol=1e-2, rtol=1.6e-2)
