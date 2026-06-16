# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Regression test for issue #270: routing weights at (num_experts=128, topk=8)
# through the patched vllm._custom_ops entry points.

import pytest
import torch

import vllm_metax.patch  # noqa: F401  # installs the topk routing patch
import vllm._custom_ops as ops

NUM_TOKENS = 64
SHAPES = [(128, 8), (128, 4), (64, 8), (256, 8)]


def _run(op, num_experts, topk, renormalize, bias):
    gating = torch.randn(NUM_TOKENS, num_experts, dtype=torch.float32, device="cuda")
    w = torch.empty(NUM_TOKENS, topk, dtype=torch.float32, device="cuda")
    idx = torch.empty(NUM_TOKENS, topk, dtype=torch.int32, device="cuda")
    tei = torch.empty(NUM_TOKENS, topk, dtype=torch.int32, device="cuda")
    op(w, idx, tei, gating, renormalize, bias)
    return gating, w, idx


def _ref(act, gating, idx, renormalize):
    scores = act(gating.float())
    w = scores.gather(1, idx.long())
    if renormalize:
        w = w / w.sum(dim=-1, keepdim=True)
    return w


@pytest.mark.parametrize("num_experts,topk", SHAPES)
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
def test_topk_softmax_weights(num_experts, topk, renormalize, with_bias):
    torch.manual_seed(0)
    bias = (
        torch.randn(num_experts, dtype=torch.float32, device="cuda")
        if with_bias
        else None
    )
    gating, w, idx = _run(ops.topk_softmax, num_experts, topk, renormalize, bias)
    ref = _ref(lambda g: torch.softmax(g, -1), gating, idx, renormalize)
    torch.testing.assert_close(w, ref, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("num_experts,topk", SHAPES)
@pytest.mark.parametrize("renormalize", [False, True])
def test_topk_sigmoid_weights(num_experts, topk, renormalize):
    torch.manual_seed(0)
    gating, w, idx = _run(ops.topk_sigmoid, num_experts, topk, renormalize, None)
    ref = _ref(torch.sigmoid, gating, idx, renormalize)
    torch.testing.assert_close(w, ref, atol=1e-5, rtol=1e-5)
