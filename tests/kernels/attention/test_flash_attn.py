# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch

from vllm_metax.v1.attention.backends.fa_utils import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    get_flash_attn_version,
)

pytestmark = pytest.mark.requires_maca


# Adapted from vLLM v0.33.0 tests/kernels/attention/test_flash_attn.py.
@pytest.mark.parametrize("paged", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_out", [False, True])
@torch.inference_mode()
def test_fa2_varlen_gqa_matches_reference(device, paged, dtype, use_out):
    torch.manual_seed(0)
    fa_version = get_flash_attn_version(head_size=64, kv_cache_dtype="auto")
    assert fa_version == 2
    q_lens, kv_lens = (3, 5), (23, 37)
    page_size, heads, dim = 16, 2, 64
    query = torch.randn(sum(q_lens), 2 * heads, dim, device=device, dtype=dtype)
    key = torch.randn(6, page_size, heads, dim, device=device, dtype=dtype)
    value = torch.randn_like(key)
    pages = [[4, 1, 0], [3, 2, 5]]
    key_rows = [
        key[row].flatten(0, 1)[:n] for row, n in zip(pages, kv_lens, strict=True)
    ]
    value_rows = [
        value[row].flatten(0, 1)[:n] for row, n in zip(pages, kv_lens, strict=True)
    ]
    if paged:
        cache_kwargs = {
            "block_table": torch.tensor(pages, device=device, dtype=torch.int32),
            "seqused_k": torch.tensor(kv_lens, device=device, dtype=torch.int32),
        }
    else:
        key, value = torch.cat(key_rows), torch.cat(value_rows)
        cache_kwargs = {
            "cu_seqlens_k": torch.tensor(
                [0, kv_lens[0], sum(kv_lens)],
                device=device,
                dtype=torch.int32,
            )
        }
    scale = dim**-0.5
    out = torch.empty_like(query) if use_out else None
    actual, lse = flash_attn_varlen_func(
        q=query,
        k=key,
        v=value,
        out=out,
        max_seqlen_q=max(q_lens),
        cu_seqlens_q=torch.tensor(
            [0, q_lens[0], sum(q_lens)], device=device, dtype=torch.int32
        ),
        max_seqlen_k=max(kv_lens),
        softmax_scale=scale,
        causal=True,
        window_size=(16, 0),
        softcap=5.0,
        return_softmax_lse=True,
        fa_version=fa_version,
        **cache_kwargs,
    )
    if use_out:
        assert actual is out

    expected, expected_lse = [], []
    for q, k, v in zip(query.split(q_lens), key_rows, value_rows, strict=True):
        logits = (
            torch.einsum(
                "qhd,khd->hqk", q.float(), k.float().repeat_interleave(2, dim=1)
            )
            * scale
        )
        logits = 5 * torch.tanh(logits / 5)
        cols = torch.arange(len(k), device=device)
        rows = cols[-len(q) :, None]
        logits = logits.masked_fill((cols > rows) | (cols < rows - 16), -torch.inf)
        expected_lse.append(logits.logsumexp(-1))
        expected.append(
            torch.einsum(
                "hqk,khd->qhd",
                logits.softmax(dim=-1),
                v.float().repeat_interleave(2, dim=1),
            )
        )
    torch.testing.assert_close(
        actual, torch.cat(expected).to(dtype), atol=1.5e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        lse, torch.cat(expected_lse, dim=1), atol=2e-3, rtol=2e-3
    )


@pytest.mark.parametrize("paged", [False, True])
@torch.inference_mode()
def test_fa2_cache_attention_appends_at_the_request_positions(device, paged):
    torch.manual_seed(0)
    page_size, heads, dim = 16, 2, 64
    lengths = [15, 27]
    table = [[3, 1], [2, 0]]
    key = torch.randn(2, 32, heads, dim, device=device, dtype=torch.float16)
    value = torch.randn_like(key)
    new_key = torch.randn(2, 2, heads, dim, device=device, dtype=key.dtype)
    new_value = torch.randn_like(new_key)
    query = torch.randn(2, 2, 4, dim, device=device, dtype=key.dtype)
    expected_key, expected_value = key.clone(), value.clone()
    for row, length in enumerate(lengths):
        expected_key[row, length : length + 2] = new_key[row]
        expected_value[row, length : length + 2] = new_value[row]

    kwargs = {}

    def to_pages(cache):
        storage = torch.empty(
            (4, page_size, heads, dim), device=device, dtype=cache.dtype
        )
        storage[torch.tensor(table, device=device).flatten()] = cache.reshape_as(
            storage
        )
        return storage

    if paged:
        key, value = to_pages(key), to_pages(value)
        kwargs["block_table"] = torch.tensor(table, device=device, dtype=torch.int32)
    out = torch.empty_like(query)
    actual = flash_attn_with_kvcache(
        query,
        key,
        value,
        k=new_key,
        v=new_value,
        cache_seqlens=torch.tensor(lengths, device=device, dtype=torch.int32),
        causal=True,
        out=out,
        fa_version=2,
        **kwargs,
    )
    assert actual is out
    torch.testing.assert_close(
        key, to_pages(expected_key) if paged else expected_key, atol=0, rtol=0
    )
    torch.testing.assert_close(
        value, to_pages(expected_value) if paged else expected_value, atol=0, rtol=0
    )
    reference = []
    for row, length in enumerate(lengths):
        k = expected_key[row, : length + 2].float().repeat_interleave(2, dim=1)
        v = expected_value[row, : length + 2].float().repeat_interleave(2, dim=1)
        logits = torch.einsum("qhd,khd->hqk", query[row].float(), k) * dim**-0.5
        positions = torch.arange(length + 2, device=device)
        logits = logits.masked_fill(positions > positions[-2:, None], -torch.inf)
        reference.append(torch.einsum("hqk,khd->qhd", logits.softmax(-1), v))
    torch.testing.assert_close(
        actual, torch.stack(reference).to(query.dtype), atol=1.5e-2, rtol=1e-2
    )
