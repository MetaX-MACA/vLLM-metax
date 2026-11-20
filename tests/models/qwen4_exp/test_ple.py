# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from itertools import accumulate

import pytest
import torch

from vllm_metax.models.qwen4_exp.common.ops.ple import ple_gate, ple_ngram_ids

pytestmark = pytest.mark.requires_maca


# Adapted from vLLM v0.33.0 tests/models/qwen4_exp/test_ple.py.
@pytest.mark.parametrize("with_output", [False, True])
@pytest.mark.parametrize("multiplier", [17, 2**62 + 17])
def test_ple_ngram_hashes_preserve_request_and_document_boundaries(
    device,
    with_output,
    multiplier,
):
    queries = [[3, 2, 17, 18], [], [19, 20], [21]]
    contexts = [[11, 12, 13], [14, 15, 16], [22, 2, 23], [24, 25, 26]]
    eos, heads_per_ngram = 2, 2
    multipliers = [multiplier, 19, 29, 31]
    sizes = [7, 11, 13, 17, 19, 23]
    offsets = [0, *accumulate(sizes[:-1])]
    output = (
        torch.empty((7, 6), device=device, dtype=torch.int64) if with_output else None
    )
    actual = ple_ngram_ids(
        torch.tensor(sum(queries, []), device=device, dtype=torch.int32),
        torch.tensor(
            [0, *accumulate(map(len, queries))], device=device, dtype=torch.int32
        ),
        torch.tensor(contexts, device=device, dtype=torch.int32),
        torch.tensor(multipliers, device=device, dtype=torch.int64),
        torch.tensor(sizes, device=device, dtype=torch.int64),
        torch.tensor(offsets, device=device, dtype=torch.int64),
        eos,
        heads_per_ngram,
        output,
    )
    if with_output:
        assert actual is output

    expected = []
    for context, query in zip(contexts, queries, strict=True):
        for i, token in enumerate(query):
            history = (context + query[:i])[-3:]
            if eos in history:
                eos_position = len(history) - 1 - history[::-1].index(eos)
                history[:eos_position] = [eos] * eos_position
            running_hash = token * multipliers[0]
            row = []
            for order, predecessor in enumerate(reversed(history)):
                running_hash ^= predecessor * multipliers[order + 1]
                signed = (running_hash + 2**63) % 2**64 - 2**63
                for index in range(
                    order * heads_per_ngram, (order + 1) * heads_per_ngram
                ):
                    row.append(offsets[index] + signed % sizes[index])
            expected.append(row)
    torch.testing.assert_close(
        actual, torch.tensor(expected, device=device), atol=0, rtol=0
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("strided", [True, False])
def test_ple_gate_matches_reference(device, dtype, strided):
    torch.manual_seed(0)
    tokens, heads, hidden = 17, 4, 96
    kv = torch.randn(tokens, (heads + 1) * hidden, device=device, dtype=dtype)
    key, value = kv.split((heads * hidden, hidden), dim=-1)
    if not strided:
        key, value = key.contiguous(), value.contiguous()
    query = torch.randn_like(key).contiguous()
    weights = [
        torch.randn(heads * hidden, device=device, dtype=dtype) / 8 for _ in range(3)
    ]

    def norm(x, weight):
        grouped = x.float().unflatten(-1, (heads, hidden))
        normalized = grouped * torch.rsqrt(
            grouped.square().mean(-1, keepdim=True) + 1e-6
        )
        return (normalized.flatten(-2) * (1 + weight.float())).to(dtype)

    gated, normed = ple_gate(key, value, query, *weights, 1e-6)
    product = norm(key, weights[0]) * norm(query, weights[1])
    dot = product.unflatten(-1, (heads, hidden)).sum(-1, keepdim=True) / hidden**0.5
    gate = torch.sigmoid(dot.sign() * dot.abs().clamp_min(1e-6).sqrt())
    expected_gated = (gate * value.unsqueeze(1)).flatten(-2)
    expected_normed = norm(expected_gated, weights[2])
    torch.testing.assert_close(gated, expected_gated, atol=0, rtol=0)
    torch.testing.assert_close(normed, expected_normed, atol=1e-2, rtol=1e-2)
