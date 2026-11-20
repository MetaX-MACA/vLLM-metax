# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch

import vllm_metax._custom_ops as mx_ops
from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopKRouter,
)
from vllm_metax.patch.enhancement.performance import grouped_topk_router as maca_router


def _single_group_reference(logits, bias, topk, *, sigmoid, renormalize, scale):
    scores = 0.5 * torch.tanh(0.5 * logits.float()) + 0.5 if sigmoid else logits.float()
    indices = torch.argsort(
        scores + bias.float(), dim=-1, descending=True, stable=True
    )[:, :topk]
    values = scores.gather(1, indices)
    if renormalize:
        values = values / (values.sum(dim=-1, keepdim=True) + 1e-20)
    return values * scale, indices.to(torch.int32)


@pytest.mark.requires_maca
@pytest.mark.parametrize("sigmoid", [False, True])
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("scale", [1.0, 2.5])
def test_grouped_topk_single_group_matches_torch_reference(
    device, sigmoid, renormalize, scale
):
    torch.manual_seed(0)
    logits = torch.randn(33, 64, device=device, dtype=torch.bfloat16)
    bias = torch.randn(64, device=device, dtype=torch.float32)
    expected_weights, expected_ids = _single_group_reference(
        logits,
        bias,
        topk=8,
        sigmoid=sigmoid,
        renormalize=renormalize,
        scale=scale,
    )
    actual_weights, actual_ids = mx_ops.grouped_topk(
        logits,
        num_expert_group=1,
        topk_group=1,
        topk=8,
        renormalize=renormalize,
        routed_scaling_factor=scale,
        bias=bias,
        scoring_func=int(sigmoid),
    )
    torch.testing.assert_close(actual_ids, expected_ids)
    torch.testing.assert_close(actual_weights, expected_weights, atol=2e-5, rtol=0)


@pytest.mark.requires_maca
def test_grouped_topk_empty_batch_preserves_plugin_contract(device):
    scores = torch.empty(0, 16, device=device, dtype=torch.bfloat16)
    bias = torch.empty(16, device=device, dtype=torch.float32)
    weights, ids = mx_ops.grouped_topk(
        scores,
        num_expert_group=4,
        topk_group=2,
        topk=2,
        renormalize=True,
        routed_scaling_factor=1.0,
        bias=bias,
    )
    assert weights.shape == (0, 2)
    assert ids.shape == (0, 2)
    assert weights.dtype == torch.float32
    assert ids.dtype == torch.int32


def test_grouped_topk_schema_is_registered_in_moe_namespace():
    schema = torch.ops._moe_C.grouped_topk.default._schema
    assert [argument.name for argument in schema.arguments] == [
        "scores",
        "n_group",
        "topk_group",
        "topk",
        "renormalize",
        "routed_scaling_factor",
        "bias",
        "scoring_func",
    ]


# Adapted from vLLM v0.33.0 tests/kernels/moe/test_grouped_topk.py.
@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("scoring", ["sigmoid", "softmax"])
@pytest.mark.parametrize("renormalize", [True, False])
@pytest.mark.parametrize(("groups", "selected_groups"), [(1, 1), (8, 4)])
def test_grouped_router_uses_unbiased_weights(
    device, monkeypatch, fused, scoring, renormalize, groups, selected_groups
):
    monkeypatch.setenv("VLLM_USE_FUSED_MOE_GROUPED_TOPK", str(int(fused)))
    torch.manual_seed(0)
    tokens, experts, top_k = 33, 256, 8
    logits = torch.randn(tokens, experts, device=device)
    hidden = torch.randn(tokens, 32, device=device)
    bias = torch.randn(experts, device=device)
    router = GroupedTopKRouter(
        top_k,
        experts,
        groups,
        selected_groups,
        renormalize,
        scoring_func=scoring,
        routed_scaling_factor=2.5,
        e_score_correction_bias=bias,
    )
    assert GroupedTopKRouter._compute_routing is maca_router._compute_routing
    weights, ids = router.select_experts(hidden, logits)

    scores = logits.sigmoid() if scoring == "sigmoid" else logits.softmax(dim=-1)
    biased = (scores + bias).reshape(tokens, groups, -1)
    group_scores = biased.topk(2, dim=-1).values.sum(-1)
    chosen_groups = group_scores.topk(selected_groups, dim=-1).indices
    group_mask = torch.zeros_like(group_scores, dtype=torch.bool)
    group_mask.scatter_(1, chosen_groups, True)
    expected_ids = (
        biased.masked_fill(~group_mask[..., None], -torch.inf)
        .flatten(1)
        .topk(top_k)
        .indices
    )
    expected_weights = scores.gather(1, expected_ids)
    if renormalize:
        expected_weights /= expected_weights.sum(dim=-1, keepdim=True)
    actual_order = ids.argsort(dim=-1)
    expected_order = expected_ids.argsort(dim=-1)
    torch.testing.assert_close(
        ids.gather(1, actual_order),
        expected_ids.gather(1, expected_order).to(torch.int32),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        weights.gather(1, actual_order),
        (expected_weights * 2.5).gather(1, expected_order),
        atol=2e-5,
        rtol=0,
    )


@pytest.mark.parametrize("skip_padding", [True, False])
def test_grouped_router_masks_graph_padding(device, monkeypatch, skip_padding):
    monkeypatch.setenv("VLLM_MOE_SKIP_PADDING", "1")
    torch.manual_seed(0)
    hidden = torch.randn(2, 16, device=device)
    logits = torch.randn(2, 896, device=device)
    padding = torch.tensor([False, True], device=device)
    router = GroupedTopKRouter(16, 896, 1, 1, skip_padding=skip_padding)
    with set_forward_context(None, VllmConfig(), is_padding=padding):
        weights, ids = router.select_experts(hidden, logits)
    assert (ids[0] >= 0).all()
    if skip_padding:
        assert (ids[1] == -1).all()
        assert (weights[1] == 0).all()
    else:
        assert (ids[1] >= 0).all()
        assert (weights[1] > 0).all()
