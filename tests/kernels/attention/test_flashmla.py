# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch

from vllm_metax.v1.attention.ops.flashmla import (
    flash_mla_with_kvcache,
    get_mla_metadata,
)

pytestmark = pytest.mark.requires_maca


@pytest.mark.parametrize("queries", [1, 2])
@pytest.mark.parametrize("heads", [16, 32])
@torch.inference_mode()
def test_flash_mla(device, queries, heads):
    rng = torch.Generator(device=device).manual_seed(0)
    rank, dim, block_size = 512, 576, 64
    lengths = [83, 39]
    pages = [[3, 0], [2, 1]]
    q = torch.randn(
        2, queries, heads, dim, device=device, dtype=torch.bfloat16, generator=rng
    )
    cache = torch.randn(
        4, block_size, 1, dim, device=device, dtype=q.dtype, generator=rng
    )
    block_table = torch.tensor(pages, device=device, dtype=torch.int32)
    seq_lens = torch.tensor(lengths, device=device, dtype=torch.int32)
    metadata, _ = get_mla_metadata(seq_lens, queries * heads, 1)
    out = torch.empty((2, queries, heads, rank), dtype=q.dtype, device=device)
    scale = dim**-0.5

    actual, lse = flash_mla_with_kvcache(
        q,
        cache,
        block_table,
        seq_lens,
        rank,
        metadata,
        softmax_scale=scale,
        causal=True,
        out=out,
    )

    refs, lse_refs = [], []
    for batch, (row, length) in enumerate(zip(pages, lengths, strict=True)):
        kv = cache[row].flatten(0, 1)[:length, 0].float()
        logits = torch.einsum("qhd,sd->hqs", q[batch].float(), kv) * scale
        keys = torch.arange(length, device=device)
        logits = logits.masked_fill(keys > keys[-queries:, None], -torch.inf)
        refs.append(torch.einsum("hqs,sd->qhd", logits.softmax(-1), kv[:, :rank]))
        lse_refs.append(logits.logsumexp(-1))
    assert actual.data_ptr() == out.data_ptr()
    torch.testing.assert_close(
        out, torch.stack(refs).to(q.dtype), atol=1e-2, rtol=1.6e-2
    )
    torch.testing.assert_close(lse, torch.stack(lse_refs), atol=2e-3, rtol=2e-3)
