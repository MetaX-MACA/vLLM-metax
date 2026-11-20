# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from itertools import accumulate

import pytest
import torch

from vllm_metax.models.deepseek_v41.metax.ops.kv_cache_int8 import (
    dequantize_and_gather_k_cache_int8,
    quantize_and_insert_k_cache_int8,
)


# Adapted from vLLM v0.33.0 tests/kernels/test_compressor_kv_cache.py for
# the MetaX block-segregated INT8 MLA layout.
@pytest.mark.requires_maca
@pytest.mark.parametrize("ratio", [1, 4])
@pytest.mark.parametrize("partial_gather", [False, True])
def test_deepseek_v41_int8_cache_rotates_quantizes_and_gathers(
    device, ratio, partial_gather
):
    torch.manual_seed(0)
    page_size, head_dim = 16, 512
    pages = [[4, 1], [3, 0]]
    lengths = [19, 18]
    slots = [
        row[i // page_size] * page_size + i % page_size
        for row, length in zip(pages, lengths, strict=True)
        for i in range(length)
    ]
    slot_mapping = torch.tensor([*slots, -1], device=device)
    states = torch.randn(
        len(slot_mapping), head_dim, device=device, dtype=torch.bfloat16
    )
    states[0, :32] = 0
    positions = torch.arange(len(states), device=device) + 1
    angles = positions.float()[:, None] * torch.linspace(0.01, 0.5, 32, device=device)
    rope_cache = torch.cat((angles.cos(), angles.sin()), dim=1)
    rope_cache = torch.cat((rope_cache[:1], rope_cache), dim=0)
    backing = torch.full((5, 2, page_size, 576), -111, device=device, dtype=torch.int8)
    cache = backing[:, 1]
    quantize_and_insert_k_cache_int8(
        states,
        cache,
        slot_mapping,
        positions,
        rope_cache,
        compress_ratio=ratio,
    )

    actual_bytes = cache.flatten(1)
    values = actual_bytes[:, : page_size * 512].reshape(-1, 512)
    scales = actual_bytes[:, page_size * 512 :].view(torch.float32).reshape(-1, 16)
    torch.testing.assert_close(
        backing[:, 0], torch.full_like(backing[:, 0], -111), atol=0, rtol=0
    )
    unused = torch.ones(values.shape[0], device=device, dtype=torch.bool)
    unused[slots] = False
    assert (values[unused] == -111).all()

    rotary = states[:-1, 448:].float()
    even, odd = rotary[:, ::2], rotary[:, 1::2]
    cos, sin = rope_cache[(positions[:-1] // ratio) * ratio].chunk(2, dim=-1)
    rotated = torch.stack(
        (even * cos - odd * sin, odd * cos + even * sin), dim=-1
    ).flatten(1)
    rows = torch.cat((states[:-1, :448].float(), rotated), dim=1).unflatten(1, (16, 32))
    expected_scales = (rows.abs().amax(-1) / 127).clamp_min(2**-30)
    scaled = rows / expected_scales.unsqueeze(-1)
    expected_codes = (scaled.sign() * (scaled.abs() + 0.5).floor()).clamp(-127, 127)
    torch.testing.assert_close(scales[slots], expected_scales, atol=1e-8, rtol=1e-6)
    torch.testing.assert_close(
        values[slots], expected_codes.flatten(1).to(torch.int8), atol=1, rtol=0
    )

    requested = [5, 7] if partial_gather else lengths
    out = torch.full(
        (2, 2 + max(requested) + 1, 512), -123, device=device, dtype=torch.bfloat16
    )
    dequantize_and_gather_k_cache_int8(
        out,
        cache,
        torch.tensor(lengths, device=device, dtype=torch.int32),
        torch.tensor(requested, device=device, dtype=torch.int32)
        if partial_gather
        else None,
        torch.tensor(pages, device=device, dtype=torch.int32),
        page_size,
        offset=2,
    )
    actual_dequantized = (
        (values[slots].float().unflatten(1, (16, 32)) * scales[slots].unsqueeze(-1))
        .flatten(1)
        .to(torch.bfloat16)
    )
    for i, (stop, count) in enumerate(zip(accumulate(lengths), requested, strict=True)):
        torch.testing.assert_close(
            out[i, 2 : 2 + count],
            actual_dequantized[stop - count : stop],
            atol=0,
            rtol=0,
        )
        assert (out[i, :2] == -123).all()
        assert (out[i, 2 + count :] == -123).all()
