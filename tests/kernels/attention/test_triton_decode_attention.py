# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

import pytest
import torch

from vllm_metax.v1.attention.ops.triton_decode_attention import decode_attention_fwd

pytestmark = pytest.mark.requires_maca


def _paged_kv(cache, block_table, length, query_heads):
    row = cache[block_table.long()].flatten(0, 1)[:length]
    return row.float().repeat_interleave(query_heads // row.shape[1], dim=1)


# Adapted from vLLM v0.33.0 tests/kernels/attention/test_triton_decode_attention.py.
@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize(
    ("kv_heads", "key_dim", "value_dim"), [(4, 64, 64), (2, 64, 64), (1, 192, 128)]
)
@pytest.mark.parametrize("logit_cap", [0.0, 5.0])
@torch.inference_mode()
def test_decode_attention_matches_torch(
    device, page_size, kv_heads, key_dim, value_dim, logit_cap
):
    torch.manual_seed(0)
    lengths = [17, 65]
    query_heads, num_splits = 4, 4
    pages_per_seq = (max(lengths) + page_size - 1) // page_size
    pages = 2 * pages_per_seq
    table = (
        torch.randperm(pages, device=device).reshape(2, pages_per_seq).to(torch.int32)
    )
    key = torch.randn(
        pages, page_size, kv_heads, key_dim, device=device, dtype=torch.bfloat16
    )
    value = torch.randn(
        pages, page_size, kv_heads, value_dim, device=device, dtype=key.dtype
    )
    query = torch.randn(2, query_heads, key_dim, device=device, dtype=key.dtype)
    actual = torch.empty(2, query_heads, value_dim, device=device, dtype=key.dtype)
    lse = torch.empty(2, query_heads, device=device, dtype=torch.float32)
    segments = torch.empty(
        (2, query_heads, num_splits, value_dim + 1), device=device, dtype=torch.float32
    )
    scale = key_dim**-0.5
    decode_attention_fwd(
        query,
        key,
        value,
        actual,
        lse,
        table,
        torch.tensor(lengths, device=device, dtype=torch.int32),
        segments,
        num_splits,
        scale,
        page_size=page_size,
        logit_cap=logit_cap,
    )

    expected, expected_lse = [], []
    for seq, length in enumerate(lengths):
        kv = _paged_kv(key, table[seq], length, query_heads)
        vv = _paged_kv(value, table[seq], length, query_heads)
        logits = torch.einsum("hd,khd->hk", query[seq].float(), kv) * scale
        if logit_cap:
            logits = logit_cap * torch.tanh(logits / logit_cap)
        expected.append(torch.einsum("hk,khd->hd", logits.softmax(dim=-1), vv))
        expected_lse.append(logits.logsumexp(dim=-1))
    torch.testing.assert_close(
        actual, torch.stack(expected).to(query.dtype), atol=4e-3, rtol=4e-3
    )
    torch.testing.assert_close(lse, torch.stack(expected_lse), atol=4e-3, rtol=4e-3)
