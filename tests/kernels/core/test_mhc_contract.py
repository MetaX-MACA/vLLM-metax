# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch

from vllm_metax.model_executor.kernels.mhc import torch as mx_torch
from vllm_metax.registry.custom_ops.layers.mhc import (
    HCHeadOp,
    MHCFusedPostPreOp,
    MHCPostOp,
    MHCPreDelayedOp,
    MHCPreOp,
)


def test_sinkhorn_reference_normalizes_both_axes(device):
    torch.manual_seed(0)
    logits = torch.randn(7, 4, 4, device=device)
    normalized = mx_torch.sinkhorn_normalize_ref(logits, repeat=5, eps=1e-6)
    assert torch.isfinite(normalized).all()
    torch.testing.assert_close(
        normalized.sum(-2), torch.ones_like(normalized[..., 0, :]), atol=2e-5, rtol=0
    )
    assert torch.all(normalized.sum(-1) > 0)


def test_mhc_pre_split_mixes_reference_shapes_and_gates(device):
    torch.manual_seed(0)
    tokens, hc_mult = 5, 4
    mixes = torch.randn(tokens, hc_mult, hc_mult * (hc_mult + 2), device=device)
    scale = torch.tensor([0.5, 0.75, 1.25], device=device)
    bias = torch.randn(hc_mult * (hc_mult + 2), device=device)
    pre, post, comb = mx_torch.mhc_pre_split_mixes_ref(
        mixes,
        scale,
        bias,
        mhc_mult=hc_mult,
        mhc_post_mult_value=2.0,
        mhc_pre_eps=1e-6,
    )
    assert pre.shape == (tokens, hc_mult, hc_mult, 1)
    assert post.shape == (tokens, hc_mult, hc_mult, 1)
    assert comb.shape == (tokens, hc_mult, hc_mult, hc_mult)
    assert torch.all(pre > 0)
    assert torch.all(post >= 0)
    assert torch.isfinite(comb).all()


def test_mhc_custom_ops_are_registered_by_plugin():
    from vllm.model_executor.custom_op import op_registry_oot

    expected = {
        "MHCPreOp": MHCPreOp,
        "MHCPreDelayedOp": MHCPreDelayedOp,
        "MHCPostOp": MHCPostOp,
        "HCHeadOp": HCHeadOp,
        "MHCFusedPostPreOp": MHCFusedPostPreOp,
    }
    for name, plugin_class in expected.items():
        assert op_registry_oot[name] is plugin_class
        assert plugin_class.forward_oot.__module__.startswith("vllm_metax.")


@pytest.mark.parametrize(
    "op_name",
    [
        "mx_mhc_pre_tilelang",
        "mx_mhc_pre_delayed_tilelang",
        "mx_mhc_post_tilelang",
        "mx_mhc_fused_post_pre_tilelang",
        "mx_hc_head_fused_kernel_tilelang",
    ],
)
def test_mhc_custom_operator_schema_registered(op_name):
    assert hasattr(torch.ops.vllm, op_name)
    assert callable(getattr(torch.ops.vllm, op_name))
