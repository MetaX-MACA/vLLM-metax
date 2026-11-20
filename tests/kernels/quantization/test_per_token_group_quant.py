# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch


from vllm_metax.platform import MacaPlatform

pytestmark = pytest.mark.requires_maca


# Adapted from vLLM v0.33.0 tests/kernels/quantization/test_per_token_group_quant.py.
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(("rows", "width"), [(0, 128), (1, 128), (7, 384), (33, 512)])
@pytest.mark.parametrize("group_size", [64, 128])
def test_native_int8_group_quant_tail_and_zero_groups(
    device, dtype, rows, width, group_size
):
    MacaPlatform.import_kernels()
    torch.manual_seed(42)
    x = torch.randn(rows, width, device=device, dtype=dtype) * 8
    x[:, :group_size] = 0
    groups = x.float().unflatten(-1, (-1, group_size))
    eps = 1e-10
    expected_scales = groups.abs().amax(-1).clamp_min(eps) / 127
    scaled = groups / expected_scales.unsqueeze(-1)
    expected_codes = scaled.round().clamp(-128, 127).to(torch.int8).flatten(1)
    codes = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty_like(expected_scales)

    torch.ops._C.per_token_group_quant_int8(
        x, codes, scales, group_size, eps, -128.0, 127.0
    )

    torch.testing.assert_close(scales, expected_scales, atol=0, rtol=1e-6)
    torch.testing.assert_close(codes, expected_codes, atol=1, rtol=0)
    assert (codes[:, :group_size] == 0).all()
