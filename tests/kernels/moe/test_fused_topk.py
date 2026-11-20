# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

import pytest
import torch

from vllm_metax.platform import MacaPlatform

pytestmark = pytest.mark.requires_maca


# Adapted from vLLM v0.33.0 tests/kernels/moe/test_fused_topk.py.
@pytest.mark.parametrize(
    ("experts", "scoring"),
    [(6, "softmax"), (6, "sigmoid"), (352, "softmax"), (257, "sigmoid")],
)
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_native_fused_topk_routes_with_bias_and_padding(
    device, experts, scoring, renormalize, dtype
):
    MacaPlatform.import_kernels()
    torch.manual_seed(0)
    tokens, topk = 33, 3
    logits = torch.randn(tokens, experts, device=device, dtype=dtype)
    bias = torch.randn(experts, device=device, dtype=torch.float32)
    padding = torch.arange(tokens, device=device) % 7 == 0
    logits[padding] = torch.nan
    weights = torch.full((tokens, topk), torch.nan, device=device)
    ids = torch.full((tokens, topk), -2, device=device, dtype=torch.int32)
    route_indices = torch.empty_like(ids)
    if scoring == "softmax":
        torch.ops._moe_C.topk_softmax(
            weights, ids, route_indices, logits, renormalize, bias, padding
        )
        scores = logits[~padding].float().softmax(-1)
        scale = 1.0
    else:
        scale = 2.5
        torch.ops._moe_C.topk_sigmoid(
            weights, ids, route_indices, logits, renormalize, bias, scale, padding
        )
        scores = logits[~padding].float().sigmoid()
    expected_ids = (scores + bias).topk(topk).indices
    expected_weights = scores.gather(1, expected_ids)
    if renormalize:
        expected_weights /= expected_weights.sum(-1, keepdim=True)
    torch.testing.assert_close(
        ids[~padding], expected_ids.to(torch.int32), atol=0, rtol=0
    )
    torch.testing.assert_close(
        weights[~padding], expected_weights * scale, atol=2e-5, rtol=0
    )
    expected_routes = torch.arange(tokens, device=device)[
        :, None
    ] + tokens * torch.arange(topk, device=device)
    torch.testing.assert_close(
        route_indices, expected_routes.to(torch.int32), atol=0, rtol=0
    )
    assert (ids[padding] == -1).all()
    assert (weights[padding] == 0).all()
