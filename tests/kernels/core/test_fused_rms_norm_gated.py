# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch

from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers import layernorm

from vllm_metax.registry.custom_ops import layernorm as maca_layernorm

pytestmark = pytest.mark.requires_maca

# From the upstream kernel tests' allclose_default.py.
ATOL = {torch.float16: 1e-3, torch.bfloat16: 1e-3, torch.float32: 1e-5}
RTOL = {torch.float16: 1e-3, torch.bfloat16: 1.6e-2, torch.float32: 1.3e-6}


@pytest.fixture(autouse=True)
def enable_custom_ops():
    config = VllmConfig(compilation_config=CompilationConfig(custom_ops=["all"]))
    with set_current_vllm_config(config):
        yield


# Adapted from vLLM v0.33.0 tests/kernels/core/test_fused_rms_norm_gated.py.
@pytest.mark.parametrize("norm_before_gate", [False, True])
@pytest.mark.parametrize("group_size", [None, 32])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_registered_gated_rms_norm_matches_reference(
    device, norm_before_gate, group_size, dtype
):
    torch.manual_seed(0)
    norm = layernorm.RMSNormGated(
        128,
        group_size=group_size,
        norm_before_gate=norm_before_gate,
        dtype=dtype,
        device=device,
    )
    assert type(norm) is maca_layernorm.MacaRMSNormGated
    assert norm._forward_method == norm.forward_oot
    x = torch.randn(7, 128, device=device, dtype=dtype)
    gate = torch.randn_like(x)
    torch.testing.assert_close(
        norm(x, gate),
        norm.forward_native(x, gate),
        atol=ATOL[dtype],
        rtol=RTOL[dtype],
    )
