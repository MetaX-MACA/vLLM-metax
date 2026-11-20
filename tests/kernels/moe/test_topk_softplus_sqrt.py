# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch
from torch.nn import functional as F

from vllm_metax.platform import MacaPlatform

pytestmark = pytest.mark.requires_maca


@pytest.mark.parametrize("route", ["biased", "hash", "mixed"])
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_fused_topk_softplus_sqrt(device, dtype, renormalize, route):
    MacaPlatform.import_kernels()
    tokens, experts, topk = 33, 128, 8
    rng = torch.Generator(device=device).manual_seed(0)
    logits = torch.randn(tokens, experts, device=device, dtype=dtype, generator=rng)
    bias = (
        torch.linspace(-1.5, 1.5, experts, device=device) if route == "biased" else None
    )
    token_ids = table = bias_vl = None
    image_sentinel = 0
    scores = F.softplus(logits.float()).sqrt()
    if route == "biased":
        expected_ids = (scores + bias).argsort(descending=True, stable=True)[:, :topk]
    else:
        token_ids = torch.arange(tokens, dtype=torch.int64, device=device)
        table = (token_ids[:, None] + 9 * torch.arange(topk, device=device)) % experts
        expected_ids = table
        if route == "mixed":
            image_sentinel = 8
            bias_vl = torch.linspace(2.5, -2.5, experts, device=device)
            image_ids = (scores + bias_vl).argsort(descending=True, stable=True)[
                :, :topk
            ]
            image_mask = (token_ids >= image_sentinel) & (
                token_ids < image_sentinel + 5
            )
            expected_ids = torch.where(image_mask[:, None], image_ids, expected_ids)
    expected_weights = scores.gather(1, expected_ids)
    if renormalize:
        expected_weights /= expected_weights.sum(-1, keepdim=True)
    factor = 1.5
    expected_weights *= factor

    weights = torch.empty(tokens, topk, dtype=torch.float32, device=device)
    ids = torch.empty_like(weights, dtype=torch.int32)
    expert_indices = torch.empty_like(ids)
    torch.ops._moe_C.topk_softplus_sqrt(
        weights,
        ids,
        expert_indices,
        logits,
        renormalize,
        factor,
        bias,
        token_ids,
        table,
        None,
        bias_vl,
        image_sentinel,
    )

    sorted_ids, order = ids.sort(dim=-1)
    sorted_expected, expected_order = expected_ids.sort(dim=-1)
    torch.testing.assert_close(
        sorted_ids, sorted_expected.to(ids.dtype), atol=0, rtol=0
    )
    torch.testing.assert_close(
        weights.gather(1, order),
        expected_weights.gather(1, expected_order),
        atol=2e-2,
        rtol=1e-2,
    )
