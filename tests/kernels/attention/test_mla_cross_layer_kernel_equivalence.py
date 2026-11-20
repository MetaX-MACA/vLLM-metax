# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch

from vllm_metax import _custom_ops as maca_ops
from vllm_metax.platform import MacaPlatform

pytestmark = pytest.mark.requires_maca


# Adapted from vLLM v0.33.0 tests/kernels/attention/test_mla_cross_layer_kernel_equivalence.py.
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_native_mla_cache_writes_only_the_selected_layer(device, dtype):
    MacaPlatform.import_kernels()
    torch.manual_seed(0)
    blocks, page_size, layers, rank, rope_dim = 7, 64, 3, 512, 64
    entries = torch.full(
        (blocks, layers, page_size, rank + rope_dim), -1, device=device, dtype=dtype
    )
    kv = torch.randn(6, rank, dtype=dtype, device=device)
    rope = torch.randn(6, rope_dim, dtype=dtype, device=device)
    slots = torch.tensor([13, 312, -1, 95, 416, 67], dtype=torch.int64, device=device)
    expected = entries.clone()
    for row, slot in enumerate(slots.tolist()):
        if slot >= 0:
            expected[slot // page_size, 1, slot % page_size] = torch.cat(
                (kv[row], rope[row])
            )

    torch.ops._C_cache_ops.concat_and_cache_mla(
        kv,
        rope,
        entries[:, 1],
        slots,
        "auto",
        torch.ones(1, device=device),
    )
    torch.testing.assert_close(entries, expected, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("lengths", [(5, 14), (37, 29)])
def test_unquantized_indexer_cache_roundtrip(device, dtype, lengths):
    torch.manual_seed(0)
    head_dim, block_size, num_blocks = 128, 16, 8
    pages = [[6, 0, 3], [5, 2, 1]]
    tokens = sum(lengths)
    data = torch.randn(tokens + 1, head_dim, device=device, dtype=dtype)
    cache = torch.full(
        (num_blocks, block_size, head_dim), -100, device=device, dtype=dtype
    )
    slots = torch.tensor(
        [
            row[i // block_size] * block_size + i % block_size
            for row, length in zip(pages, lengths, strict=True)
            for i in range(length)
        ]
        + [-1],
        device=device,
        dtype=torch.int64,
    )
    maca_ops.indexer_k_quant_and_cache(data, cache, slots, head_dim, "auto")

    expected_cache = torch.full_like(cache, -100)
    expected_cache.view(-1, head_dim)[slots[:-1]] = data[:-1]
    torch.testing.assert_close(cache, expected_cache, atol=0, rtol=0)

    output = torch.empty(tokens, 2 * head_dim, device=device, dtype=dtype)[:, :head_dim]
    maca_ops.cp_gather_indexer_k_quant_cache(
        cache.view(torch.uint8),
        output.view(torch.uint8),
        None,
        torch.tensor(pages, device=device, dtype=torch.int32),
        torch.tensor([0, lengths[0], tokens], device=device, dtype=torch.int32),
    )
    torch.testing.assert_close(output, data[:-1], atol=0, rtol=0)


@pytest.mark.parametrize("lengths", [(7, 13), (37, 31)])
def test_packed_indexer_gather_preserves_values_and_scale_bytes(device, lengths):
    MacaPlatform.import_kernels()
    blocks, block_size, head, scale_width = 8, 16, 128, 4
    pages = [[7, 2, 0], [5, 3, 1]]
    rng = torch.Generator(device=device).manual_seed(0)
    values = torch.randint(
        256, (blocks, block_size, head), dtype=torch.uint8, device=device, generator=rng
    )
    scales = torch.randint(
        256,
        (blocks, block_size, scale_width),
        dtype=torch.uint8,
        device=device,
        generator=rng,
    )
    combined = torch.cat((values.flatten(1), scales.flatten(1)), dim=1).view(
        blocks, block_size, head + scale_width
    )
    tokens = sum(lengths)
    out = torch.full((tokens + 3, head), 63, device=device, dtype=torch.uint8)
    out_scales = torch.full(
        (tokens + 3, scale_width), 37, device=device, dtype=torch.uint8
    )
    table = torch.tensor(pages, dtype=torch.int32, device=device)
    cu_seq_lens = torch.tensor(
        [0, lengths[0], tokens], dtype=torch.int32, device=device
    )

    torch.ops._C_cache_ops.cp_gather_indexer_k_quant_cache(
        combined, out, out_scales, table, cu_seq_lens
    )

    for output, src in ((out, values), (out_scales, scales)):
        expected = torch.cat(
            [
                src[p].flatten(0, 1)[:length]
                for p, length in zip(pages, lengths, strict=True)
            ]
        )
        torch.testing.assert_close(output[:tokens], expected, atol=0, rtol=0)
    assert (out[tokens:] == 63).all()
    assert (out_scales[tokens:] == 37).all()
