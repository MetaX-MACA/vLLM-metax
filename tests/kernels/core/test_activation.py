# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch

from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers import activation

from vllm_metax.registry.custom_ops import activation as maca_activation
import vllm_metax._custom_ops as mx_ops

pytestmark = pytest.mark.requires_maca

# From the upstream kernel tests' allclose_default.py.
ATOL = {torch.float16: 1e-3, torch.bfloat16: 1e-3, torch.float32: 1e-5}
RTOL = {torch.float16: 1e-3, torch.bfloat16: 1.6e-2, torch.float32: 1.3e-6}


@pytest.fixture(autouse=True)
def enable_custom_ops():
    config = VllmConfig(compilation_config=CompilationConfig(custom_ops=["all"]))
    with set_current_vllm_config(config):
        yield


# Adapted from vLLM v0.33.0 tests/kernels/core/test_activation.py.
@pytest.mark.parametrize(
    ("layer_cls", "plugin_cls", "kwargs", "gated"),
    [
        (activation.SiluAndMul, maca_activation.MacaSiluAndMul, {}, True),
        (activation.MulAndSilu, maca_activation.MacaMulAndSilu, {}, True),
        (activation.GeluAndMul, maca_activation.MacaGeluAndMul, {}, True),
        (
            activation.GeluAndMul,
            maca_activation.MacaGeluAndMul,
            {"approximate": "tanh"},
            True,
        ),
        (
            activation.FatreluAndMul,
            maca_activation.MacaFatreluAndMul,
            {"threshold": 0.5},
            True,
        ),
        (activation.SwigluOAIAndMul, maca_activation.MacaSwigluOAIAndMul, {}, True),
        (activation.SwigluStepAndMul, maca_activation.MacaSwigluStepAndMul, {}, True),
        (
            activation.SiluAndMulWithClamp,
            maca_activation.MacaSiluAndMulWithClamp,
            {"swiglu_limit": 2.0, "alpha": 1.25, "beta": 0.5},
            True,
        ),
        (
            activation.SituAndMul,
            maca_activation.MacaSituAndMul,
            {"beta": 1.5, "linear_beta": 0.75},
            True,
        ),
        (activation.FastGELU, maca_activation.MacaFastGELU, {}, False),
        (activation.NewGELU, maca_activation.MacaNewGELU, {}, False),
        (activation.QuickGELU, maca_activation.MacaQuickGELU, {}, False),
        (
            activation.ReLUSquaredActivation,
            maca_activation.MacaReLUSquaredActivation,
            {},
            False,
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens", [1, 7])
@torch.inference_mode()
def test_registered_activation_matches_reference(
    device, layer_cls, plugin_cls, kwargs, gated, dtype, tokens
):
    torch.manual_seed(0)
    layer = layer_cls(**kwargs)
    assert type(layer) is plugin_cls
    assert layer._forward_method == layer.forward_oot
    x = torch.randn(tokens, 256 if gated else 128, device=device, dtype=dtype)
    if "swiglu_limit" in kwargs:
        x *= 3
    actual = layer(x)
    expected = layer.forward_native(x)
    rtol = RTOL[dtype]
    atol = ATOL[dtype]
    if layer_cls in (activation.SwigluOAIAndMul, activation.SwigluStepAndMul):
        rtol = {torch.float16: 2e-3, torch.bfloat16: 2e-2, torch.float32: 1.3e-6}[dtype]
        atol = {torch.float16: 1e-3, torch.bfloat16: 1e-2, torch.float32: 1e-6}[dtype]
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.requires_maca
def test_swiglustep_alpha_beta_boundary_in_source_mode(monkeypatch, device):
    monkeypatch.setattr(mx_ops.mx_envs, "USE_PRECOMPILED_KERNEL", False)
    out = torch.empty(2, 8, device=device, dtype=torch.bfloat16)
    inp = torch.randn(2, 16, device=device, dtype=torch.bfloat16)
    mx_ops.swiglustep_and_mul(out, inp, clamp_limit=7.0, alpha=1.0, beta=0.0)
    assert torch.isfinite(out).all()
    with pytest.raises(NotImplementedError, match="alpha=1.0 and beta=0.0"):
        mx_ops.swiglustep_and_mul(out, inp, alpha=2.0)


@pytest.mark.requires_maca
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_source_silu_and_mul_matches_torch(device, dtype):
    x = torch.randn(7, 64, device=device, dtype=dtype)
    out = torch.empty(7, 32, device=device, dtype=dtype)
    torch.ops._C.silu_and_mul(out, x)
    reference = torch.nn.functional.silu(x[..., :32]) * x[..., 32:]
    torch.testing.assert_close(out, reference, atol=0, rtol=0)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_masked_moe_activation_dispatch(device, dtype, batched):
    counts = [0, 1, 5, 7] if batched else [3]
    hidden, tokens = 131, 7
    data = torch.randn(len(counts), tokens, hidden * 2, dtype=dtype, device=device)
    out = torch.full((len(counts), tokens, hidden), -7, dtype=dtype, device=device)
    input_tensor = data if batched else data[0]
    output_tensor = out if batched else out[0]
    lengths = torch.tensor(counts, dtype=torch.int32, device=device)
    gate, up = data.float().chunk(2, dim=-1)
    value = (torch.nn.functional.silu(gate) * up).to(dtype)
    valid = torch.arange(tokens, device=device)[None, :] < lengths[:, None]
    expected = torch.where(valid[..., None], value, -7)

    torch.ops._C.masked_moe_activation(output_tensor, input_tensor, lengths, "silu")

    torch.testing.assert_close(out, expected, atol=1e-3, rtol=1.6e-2)
    assert (out[~valid] == -7).all()
