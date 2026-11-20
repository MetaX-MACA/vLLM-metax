# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch

from vllm_metax.platform import MacaPlatform


@pytest.mark.requires_maca
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_source_rms_norm_matches_torch(device, dtype):
    MacaPlatform.import_kernels()
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
    MacaPlatform.import_kernels()
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
