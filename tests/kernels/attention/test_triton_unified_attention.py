# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

import pytest
import torch

from vllm_metax.v1.attention.ops.triton_unified_attention import unified_attention

pytestmark = pytest.mark.requires_maca


def _paged_kv(cache, block_table, length, query_heads):
    row = cache[block_table.long()].flatten(0, 1)[:length]
    return row.float().repeat_interleave(query_heads // row.shape[1], dim=1)


# Adapted from vLLM v0.33.0 tests/kernels/attention/test_triton_unified_attention.py.
@pytest.mark.parametrize("head_size", [64, 512])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize(
    ("query_lens", "window", "soft_cap", "split_k"),
    [
        ((3, 2), None, 0.0, False),
        ((3, 2), 17, 5.0, False),
        ((1, 1), None, 0.0, True),
        ((2, 2), 17, 5.0, True),
    ],
)
@torch.inference_mode()
def test_paged_attention_matches_torch(
    device, head_size, layout, query_lens, window, soft_cap, split_k
):
    torch.manual_seed(0)
    seq_lens = [19, 73]
    page_size, kv_heads, query_heads = 16, 2, 4
    block_table = torch.tensor(
        [[7, 1, 0, 0, 0], [3, 8, 5, 2, 6]], device=device, dtype=torch.int32
    )
    cache_shape = (9, page_size, kv_heads, head_size)
    if layout == "HND":
        cache_shape = (9, kv_heads, page_size, head_size)
    key, value = [
        torch.randn(cache_shape, device=device, dtype=torch.bfloat16) for _ in range(2)
    ]
    if layout == "HND":
        key, value = key.transpose(1, 2), value.transpose(1, 2)
    query = torch.randn(
        sum(query_lens), query_heads, head_size, device=device, dtype=key.dtype
    )
    actual = torch.empty_like(query)
    scale = head_size**-0.5
    scratch = {}
    if split_k:
        scratch_shape = (query.shape[0], query_heads, 4)
        scratch = {
            "seq_threshold_3D": 8,
            "num_par_softmax_segments": scratch_shape[-1],
            "softmax_segm_output": torch.empty(
                (*scratch_shape, head_size), device=device, dtype=torch.float32
            ),
            "softmax_segm_max": torch.full(
                scratch_shape, torch.nan, device=device, dtype=torch.float32
            ),
            "softmax_segm_expsum": torch.empty(
                scratch_shape, device=device, dtype=torch.float32
            ),
        }
    unified_attention(
        q=query,
        k=key,
        v=value,
        out=actual,
        cu_seqlens_q=torch.tensor(
            [0, query_lens[0], sum(query_lens)], device=device, dtype=torch.int32
        ),
        max_seqlen_q=max(query_lens),
        seqused_k=torch.tensor(seq_lens, device=device, dtype=torch.int32),
        max_seqlen_k=max(seq_lens),
        softmax_scale=scale,
        causal=True,
        window_size=(-1, -1) if window is None else (window - 1, 0),
        block_table=block_table,
        softcap=soft_cap,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        **scratch,
    )

    if split_k:
        maxima = scratch["softmax_segm_max"]
        assert not maxima[..., 0].isnan().any()
        assert maxima.isfinite().any(dim=-1).all()
    references = []
    for seq, query_part in enumerate(query.split(query_lens)):
        seq_len = seq_lens[seq]
        kv = _paged_kv(key, block_table[seq], seq_len, query_heads)
        vv = _paged_kv(value, block_table[seq], seq_len, query_heads)
        logits = torch.einsum("qhd,khd->hqk", query_part.float(), kv) * scale
        if soft_cap:
            logits = soft_cap * torch.tanh(logits / soft_cap)
        key_positions = torch.arange(seq_len, device=device)
        query_positions = key_positions[-len(query_part) :, None]
        mask = key_positions <= query_positions
        if window is not None:
            mask &= key_positions > query_positions - window
        probabilities = logits.masked_fill(~mask, -torch.inf).softmax(dim=-1)
        references.append(torch.einsum("hqk,khd->qhd", probabilities, vv))
    torch.testing.assert_close(
        actual, torch.cat(references).to(query.dtype), atol=4e-3, rtol=4e-3
    )
