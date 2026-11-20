# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch

from vllm import _custom_ops as ops

from vllm_metax.platform import MacaPlatform

pytestmark = pytest.mark.requires_maca

DTYPES = [torch.bfloat16, torch.float32]
HIDDEN_SIZES = [17, 1024, 1025, 1026, 5137, 8193]
NUM_TOKENS = [1, 7, 4096]
SCALES = [0.1, 2.1]


def _quant_input(device, num_tokens, hidden_size, dtype, asymmetric=False):
    rng = torch.Generator(device=device).manual_seed(0)
    x = (
        torch.rand(num_tokens, hidden_size, dtype=dtype, device=device, generator=rng)
        * 1000
    )
    return x - 300 if asymmetric else x


def _assert_quantized(result, x, scale, zero_point, atol):
    expected = (x.float() / scale).round()
    if zero_point is not None:
        expected += zero_point
    limits = torch.iinfo(torch.int8)
    torch.testing.assert_close(
        result, expected.clamp(limits.min, limits.max).to(torch.int8), atol=atol, rtol=0
    )


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_dynamic_scaled_int8_quant(device, num_tokens, hidden_size, dtype):
    MacaPlatform.import_kernels()
    x = _quant_input(device, num_tokens, hidden_size, dtype)
    expected_scale = x.float().abs().amax(-1, keepdim=True) / 127

    result, scale, zero_point = ops.scaled_int8_quant(x)

    assert zero_point is None
    torch.testing.assert_close(scale, expected_scale)
    _assert_quantized(result, x, expected_scale, None, atol=1)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_dynamic_scaled_int8_azp_quant(device, num_tokens, hidden_size, dtype):
    MacaPlatform.import_kernels()
    x = _quant_input(device, num_tokens, hidden_size, dtype, asymmetric=True)
    low = x.float().amin(-1, keepdim=True)
    high = x.float().amax(-1, keepdim=True)
    expected_scale = (high - low) / 255
    expected_zp = (-128 - low / expected_scale).round().to(torch.int32)

    result, scale, zero_point = ops.scaled_int8_quant(x, symmetric=False)

    torch.testing.assert_close(scale, expected_scale)
    torch.testing.assert_close(zero_point, expected_zp, atol=1, rtol=0)
    _assert_quantized(result, x, expected_scale, expected_zp, atol=2)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scale_value", SCALES)
@torch.inference_mode()
def test_static_scaled_int8_quant(device, num_tokens, hidden_size, dtype, scale_value):
    MacaPlatform.import_kernels()
    x = _quant_input(device, num_tokens, hidden_size, dtype)
    scale = torch.tensor([scale_value], device=device, dtype=torch.float32)

    result, actual_scale, zero_point = ops.scaled_int8_quant(x, scale)

    assert actual_scale is scale
    assert zero_point is None
    _assert_quantized(result, x, scale, None, atol=1)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("scale_value", SCALES)
@pytest.mark.parametrize("azp", [-255, 54])
@torch.inference_mode()
def test_static_scaled_int8_azp_quant(
    device, num_tokens, hidden_size, dtype, scale_value, azp
):
    MacaPlatform.import_kernels()
    x = _quant_input(device, num_tokens, hidden_size, dtype, asymmetric=True)
    scale = torch.tensor([scale_value], device=device, dtype=torch.float32)
    zero_point = torch.tensor([azp], device=device, dtype=torch.int32)

    result, actual_scale, actual_zp = ops.scaled_int8_quant(
        x, scale, zero_point, symmetric=False
    )

    assert actual_scale is scale
    assert actual_zp is zero_point
    _assert_quantized(result, x, scale, zero_point, atol=1)


@pytest.mark.parametrize("is_max", [True, False])
@torch.inference_mode()
def test_static_scaled_int8_azp_quant_saturating_cast(device, is_max):
    MacaPlatform.import_kernels()
    i32 = torch.iinfo(torch.int32)
    bound = torch.tensor(
        i32.max if is_max else i32.min, dtype=torch.float32, device=device
    )
    below = torch.nextafter(bound, torch.full_like(bound, -torch.inf))
    above = torch.nextafter(bound, torch.full_like(bound, torch.inf))
    x = torch.stack((below, bound, above)).unsqueeze(0)
    scale = torch.scalar_tensor(1, dtype=torch.float32, device=device)
    zero_point = torch.scalar_tensor(0, dtype=torch.int32, device=device)

    result, _, _ = ops.scaled_int8_quant(x, scale, zero_point, symmetric=False)

    i8 = torch.iinfo(torch.int8)
    expected = torch.full_like(result, i8.max if is_max else i8.min)
    torch.testing.assert_close(result, expected, atol=0, rtol=0)


@pytest.mark.parametrize("static", [False, True])
@pytest.mark.parametrize("symmetric", [False, True])
def test_scaled_int8_quant_opcheck(device, static, symmetric):
    MacaPlatform.import_kernels()
    x = _quant_input(device, 7, 17, torch.float32, asymmetric=not symmetric)
    result = torch.empty_like(x, dtype=torch.int8)
    scales = torch.full((1,) if static else (7, 1), 0.1, device=device)
    zero_points = None if symmetric else torch.zeros_like(scales, dtype=torch.int32)
    op = (
        torch.ops._C.static_scaled_int8_quant
        if static
        else torch.ops._C.dynamic_scaled_int8_quant
    )
    torch.library.opcheck(op, (result, x, scales, zero_points))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width", [64, 131])
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("static", [False, True])
def test_source_int8_quantization_matches_reference(
    device, dtype, width, symmetric, static
):
    MacaPlatform.import_kernels()
    torch.manual_seed(0)
    x = torch.randn(7, width, device=device, dtype=dtype) * 10 + 2
    x_float = x.float()
    scale = zero_point = None
    if static:
        scale = torch.tensor([0.125], device=device)
        if not symmetric:
            zero_point = torch.tensor([-13], dtype=torch.int32, device=device)
        expected_scale = scale
        expected_zp = zero_point
    elif symmetric:
        expected_scale = x_float.abs().amax(-1, keepdim=True) / 127
        expected_zp = None
    else:
        low = x_float.amin(-1, keepdim=True)
        high = x_float.amax(-1, keepdim=True)
        expected_scale = (high - low) / 255
        expected_zp = (-128 - low / expected_scale).round().to(torch.int32)

    result, actual_scale, actual_zp = ops.scaled_int8_quant(
        x,
        scale,
        zero_point,
        symmetric=symmetric,
    )
    if expected_zp is not None:
        torch.testing.assert_close(
            actual_zp, expected_zp, atol=0 if static else 1, rtol=0
        )
    else:
        assert actual_zp is None
    torch.testing.assert_close(actual_scale, expected_scale)
    # The device uses reciprocal multiplication. Near a half-integer the
    # rounded INT8 result can differ by one, plus one from the dynamic AZP.
    atol = 0 if static else 1 if symmetric else 2
    _assert_quantized(result, x, expected_scale, expected_zp, atol)
