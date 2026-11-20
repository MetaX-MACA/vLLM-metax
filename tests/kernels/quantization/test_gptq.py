# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode


from vllm_metax import _custom_ops as maca_ops
from vllm_metax.registry.linear_kernels.mp import _apply_gptq_impl, _gptq_shuffle

pytestmark = pytest.mark.requires_maca


# Adapted from vLLM v0.33.0 tests/kernels/quantization/test_gptq.py.
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gptq_fake_dispatch_uses_metax_workspace_arguments(device, dtype):
    with FakeTensorMode():
        a = torch.empty((3, 256), device=device, dtype=dtype)
        weight = torch.empty((32, 128), device=device, dtype=torch.int32)
        zeros = torch.empty((2, 16), device=device, dtype=torch.int32)
        scales = torch.empty((2, 128), device=device, dtype=dtype)
        group_index = torch.empty((256,), device=device, dtype=torch.int32)
        perm = torch.empty_like(group_index)
        workspace = torch.empty((3, 128), device=device, dtype=torch.float32)
        out = maca_ops.gptq_gemm(
            a,
            weight,
            zeros,
            scales,
            group_index,
            True,
            4,
            128,
            perm,
            workspace,
            dtype == torch.bfloat16,
        )
        assert out.shape == (3, 128)
        assert out.dtype == a.dtype
        assert out.device == a.device


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("batch_shape", [(1,), (2, 3)])
def test_gptq_shuffled_weight_and_workspace_match_dense_matmul(
    device, dtype, batch_shape
):
    # GPTQ v1 encodes zero points one below their effective value.
    torch.manual_seed(0)
    in_features, out_features, group_size = 256, 128, 128
    weight = torch.randint(
        0, 16, (in_features, out_features), device=device, dtype=torch.int32
    )
    shifts = torch.arange(0, 32, 4, device=device, dtype=torch.int32)
    packed = (weight.unflatten(0, (-1, 8)) << shifts[None, :, None]).sum(dim=1).int()
    zeros = torch.full(
        (2, out_features // 8), 0x77777777, device=device, dtype=torch.int32
    )
    scales = torch.randint(1, 5, (2, out_features), device=device).to(dtype) / 32
    # Dyadic activations isolate packing and workspace errors from accumulator
    # precision in the FP16 GEMV path.
    x = torch.randint(-8, 9, (*batch_shape, in_features), device=device).to(dtype) / 16
    g_idx = torch.empty(0, device=device, dtype=torch.int32)
    _gptq_shuffle(packed, 4)
    actual = _apply_gptq_impl(
        x,
        packed,
        scales,
        zeros,
        None,
        g_idx,
        True,
        4,
        group_size,
        False,
    )
    dequantized = (weight.float() - 8) * scales.float().repeat_interleave(
        group_size, dim=0
    )
    expected = (x.float() @ dequantized).to(dtype)
    eps = torch.finfo(dtype).eps
    torch.testing.assert_close(actual, expected, atol=eps, rtol=eps)
