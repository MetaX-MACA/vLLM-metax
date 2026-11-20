# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch


from vllm_metax import _custom_ops as maca_ops

pytestmark = pytest.mark.requires_maca


def _pack_awq(values):
    shifts = torch.tensor([0, 16, 4, 20, 8, 24, 12, 28], device=values.device)
    return (values.unflatten(-1, (-1, 8)) << shifts).sum(-1).to(torch.int32)


# Adapted from vLLM v0.33.0 tests/kernels/quantization/test_awq_triton.py.
@pytest.mark.parametrize("tokens", [1, 17])
def test_awq_repacked_bf16_weights_match_dense_matmul(device, monkeypatch, tokens):
    monkeypatch.setenv("VLLM_USE_TRITON_AWQ", "0")
    torch.manual_seed(0)
    dtype = torch.bfloat16
    in_features, out_features, group_size = 256, 128, 128
    weight = torch.randint(0, 16, (in_features, out_features), device=device)
    zeros = torch.randint(1, 15, (2, out_features), device=device)
    scales = torch.rand(2, out_features, device=device, dtype=dtype) / 16
    input = torch.randn(tokens, in_features, device=device, dtype=dtype)
    repacked = maca_ops.awq_to_gptq_4bit(_pack_awq(weight))
    assert repacked.shape == (out_features, in_features // 8)
    workspace = torch.zeros(tokens, out_features, device=device, dtype=torch.float32)
    actual = maca_ops.awq_gemm(
        input,
        repacked,
        _pack_awq(zeros),
        scales,
        8,
        workspace,
        dtype == torch.bfloat16,
    )
    dequantized = (
        (weight - zeros.repeat_interleave(group_size, dim=0)).float()
        * scales.repeat_interleave(group_size, dim=0).float()
    ).to(dtype)
    expected = (input.float() @ dequantized.float()).to(dtype)
    eps = torch.finfo(dtype).eps
    torch.testing.assert_close(actual, expected, atol=2 * eps, rtol=eps)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_dequantize(device, dtype):
    from vllm_metax.platform import MacaPlatform

    MacaPlatform.import_kernels()
    rng = torch.Generator(device=device).manual_seed(0)
    weight = torch.randint(
        16, (256, 128), device=device, dtype=torch.int32, generator=rng
    )
    zeros = torch.randint(16, (2, 128), device=device, dtype=torch.int32, generator=rng)
    scales = torch.rand(2, 128, dtype=dtype, device=device, generator=rng)
    expected = (
        (weight - zeros.repeat_interleave(128, dim=0)).float()
        * scales.repeat_interleave(128, dim=0).float()
    ).to(dtype)

    actual = torch.ops._C.awq_dequantize(
        _pack_awq(weight), scales, _pack_awq(zeros), 0, 0, 0
    )

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
