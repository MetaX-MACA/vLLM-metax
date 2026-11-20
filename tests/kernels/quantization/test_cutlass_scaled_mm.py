# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch

from vllm.model_executor.kernels.linear import init_int8_linear_kernel

from vllm_metax.registry.linear_kernels.int8 import MctlassScaledMMLinearKernel

pytestmark = pytest.mark.requires_maca


@pytest.mark.parametrize("tokens", [1, 17])
@pytest.mark.parametrize(
    ("symmetric", "with_bias"),
    [(True, False), (True, True), (False, True)],
    ids=["symmetric", "symmetric-bias", "asymmetric-bias"],
)
@torch.inference_mode()
def test_registered_int8_linear_matches_dequantized_matmul(
    device, tokens, symmetric, with_bias
):
    torch.manual_seed(0)
    dtype = torch.bfloat16
    in_features, out_features = 128, 64
    kernel = init_int8_linear_kernel(
        is_channelwise=True,
        is_static_input_scheme=False,
        input_symmetric=symmetric,
        module_name="metax-test",
    )
    assert type(kernel) is MctlassScaledMMLinearKernel
    layer = torch.nn.Module()
    layer.logical_widths = [out_features]
    weight = torch.randint(
        -8, 8, (out_features, in_features), device=device, dtype=torch.int8
    )
    weight_scale = (
        0.0625 * 2.0 ** (torch.arange(out_features, device=device) % 3)
    ).unsqueeze(1)
    layer.register_parameter("weight", torch.nn.Parameter(weight, requires_grad=False))
    layer.register_parameter(
        "weight_scale", torch.nn.Parameter(weight_scale, requires_grad=False)
    )
    kernel.process_weights_after_loading(layer)
    assert layer.weight.shape == (in_features, out_features)
    assert layer.weight.stride() == (1, in_features)

    x = torch.randint(-40, 40, (tokens, in_features), device=device).to(dtype) / 8
    x[:, 0] = -119 / 8 if not symmetric else -127 / 8
    x[:, 1] = 136 / 8 if not symmetric else 127 / 8
    bias = (
        torch.randint(-8, 9, (out_features,), device=device).to(dtype) / 8
        if with_bias
        else None
    )
    actual = kernel.apply_weights(layer, x, bias)

    zero_point = 0 if symmetric else -9
    x_quant = ((x.float() * 8).round() + zero_point).clamp(-128, 127)
    x_dequant = (x_quant - zero_point) / 8
    expected = (x_dequant @ weight.float().T) * weight_scale.T
    if bias is not None:
        expected += bias.float()
    eps = torch.finfo(dtype).eps
    torch.testing.assert_close(actual, expected.to(dtype), atol=eps, rtol=eps)


def test_cutlass_capability_queries_for_unsupported_paths(device):
    from vllm_metax.platform import MacaPlatform

    MacaPlatform.import_kernels()
    assert not torch.ops._C.cutlass_group_gemm_supported(100)
    assert not torch.ops._C.cutlass_scaled_mm_supports_block_fp8(100)
    assert not torch.ops._C.cutlass_scaled_mm_supports_fp8(0)
    assert not torch.ops._C.cutlass_scaled_mm_supports_fp4(0)
