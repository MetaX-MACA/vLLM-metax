# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch

from vllm_metax.platform import MacaPlatform

pytestmark = pytest.mark.requires_maca


@pytest.mark.parametrize("decode", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_concat_ignores_negative_slots_for_cache(device, dtype, decode):
    MacaPlatform.import_kernels()
    rng = torch.Generator(device=device).manual_seed(0)

    def values(*shape):
        return torch.randn(*shape, dtype=dtype, device=device, generator=rng)

    tokens, heads, block_size = 4, 2, 8
    kv, k_pe = values(tokens, 512), values(tokens, 96)[:, :64]
    slots = torch.tensor([-1, 0, -1, 13], dtype=torch.int64, device=device)
    cache = torch.full((2, block_size, 576), -7, dtype=dtype, device=device)
    expected = torch.full_like(cache, -7)
    expected.flatten(0, 1)[[0, 13]] = torch.cat((kv[[1, 3]], k_pe[[1, 3]]), dim=-1)

    if decode:
        ql, q_pe = values(tokens, heads, 512), values(tokens, heads, 64)
        out = torch.empty((tokens, heads, 576), dtype=dtype, device=device)
        torch.ops._C.fused_kimi_k3_mla_decode_q_concat_kv_cache_insert(
            ql, q_pe, kv, k_pe, out, cache, slots, block_size
        )
        expected_out = torch.cat((ql, q_pe), dim=-1)
    else:
        q, k = values(tokens, heads, 192), values(tokens, heads, 128)
        original_q = q.clone()
        out = torch.empty_like(q)
        torch.ops._C.fused_kimi_k3_mla_key_concat_kv_cache_insert(
            q, k, k_pe, kv, out, cache, slots, block_size
        )
        expected_out = torch.cat((k, k_pe[:, None].expand(-1, heads, -1)), dim=-1)
        torch.testing.assert_close(q, original_q, atol=0, rtol=0)
        kv_only = torch.empty_like(out)
        torch.ops._C.fused_kimi_k3_mla_kv_concat(k, k_pe, kv_only)
        torch.testing.assert_close(kv_only, expected_out, atol=0, rtol=0)

    torch.testing.assert_close(out, expected_out, atol=0, rtol=0)
    torch.testing.assert_close(cache, expected, atol=0, rtol=0)
