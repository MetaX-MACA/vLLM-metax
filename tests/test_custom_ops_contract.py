# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch

import vllm_metax._custom_ops as mx_ops
from vllm_metax._dummy_ops import dummy_ops


def _single_group_reference(logits, bias, topk, *, sigmoid, renormalize, scale):
    scores = (
        0.5 * torch.tanh(0.5 * logits.float()) + 0.5
        if sigmoid
        else logits.float()
    )
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


@pytest.mark.requires_maca
def test_swiglustep_alpha_beta_boundary_in_source_mode(monkeypatch, device):
    monkeypatch.setattr(mx_ops.mx_envs, "USE_PRECOMPILED_KERNEL", False)
    out = torch.empty(2, 8, device=device, dtype=torch.bfloat16)
    inp = torch.randn(2, 16, device=device, dtype=torch.bfloat16)
    mx_ops.swiglustep_and_mul(out, inp, clamp_limit=7.0, alpha=1.0, beta=0.0)
    assert torch.isfinite(out).all()
    with pytest.raises(NotImplementedError, match="alpha=1.0 and beta=0.0"):
        mx_ops.swiglustep_and_mul(out, inp, alpha=2.0)


def test_gptq_schema_exposes_metax_workspace_arguments():
    schema = torch.ops._C.gptq_gemm.default._schema
    argument_names = [argument.name for argument in schema.arguments]
    assert argument_names[-3:] == ["perm_space", "temp_space", "dtype_bf16"]


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


def test_dummy_scheduler_metadata_contract():
    assert (
        dummy_ops.get_scheduler_metadata(
            batch_size=1,
            max_seqlen_q=1,
            max_seqlen_k=1,
            num_heads_q=1,
            num_heads_kv=1,
            headdim=64,
            cache_seqlens=torch.zeros(1, dtype=torch.int32),
        )
        is None
    )


@pytest.mark.requires_maca
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_source_silu_and_mul_matches_torch(device, dtype):
    x = torch.randn(7, 64, device=device, dtype=dtype)
    out = torch.empty(7, 32, device=device, dtype=dtype)
    torch.ops._C.silu_and_mul(out, x)
    reference = torch.nn.functional.silu(x[..., :32]) * x[..., 32:]
    torch.testing.assert_close(out, reference, atol=0, rtol=0)


@pytest.mark.requires_maca
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_source_rms_norm_matches_torch(device, dtype):
    x = torch.randn(7, 64, device=device, dtype=dtype)
    weight = torch.randn(64, device=device, dtype=dtype)
    out = torch.empty_like(x)
    torch.ops._C.rms_norm(out, x, weight, 1e-6)
    reference = (
        x.float()
        * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
        * weight.float()
    ).to(dtype)
    torch.testing.assert_close(out, reference, atol=0, rtol=0)


@pytest.mark.requires_maca
def test_source_fused_add_rms_norm_updates_both_outputs(device):
    x = torch.randn(7, 64, device=device, dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(64, device=device, dtype=torch.bfloat16)
    expected_sum = (x.float() + residual.float()).to(torch.bfloat16)
    expected_norm = (
        expected_sum.float()
        * torch.rsqrt(expected_sum.float().square().mean(-1, keepdim=True) + 1e-6)
        * weight.float()
    ).to(torch.bfloat16)
    actual_x = x.clone()
    actual_residual = residual.clone()
    torch.ops._C.fused_add_rms_norm(actual_x, actual_residual, weight, 1e-6)
    torch.testing.assert_close(actual_residual, expected_sum, atol=0, rtol=0)
    torch.testing.assert_close(actual_x, expected_norm, atol=0, rtol=0)
