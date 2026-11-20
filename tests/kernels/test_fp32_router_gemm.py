# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch

from vllm_metax.models.deepseek_v4.metax.gate_linear import (
    _maca_fp32_router_gemm_dispatch,
)
from vllm_metax.platform import MacaPlatform

pytestmark = pytest.mark.requires_maca

SHAPES = [(3072, 256), (6144, 128), (6144, 256), (5120, 256), (4096, 352), (8192, 256)]


@pytest.mark.parametrize(
    "hidden,experts,tokens",
    [(h, e, t) for h, e in SHAPES for t in (0, 1, 17, 32)] + [(5120, 256, 48)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fp32_router_gemm(device, hidden, experts, dtype, tokens):
    MacaPlatform.import_kernels()
    rng = torch.Generator(device=device).manual_seed(42)
    activations = torch.randn(tokens, hidden, dtype=dtype, device=device, generator=rng)
    weight = torch.randn(
        experts, hidden, dtype=torch.float32, device=device, generator=rng
    )
    expected = (activations.cpu().double() @ weight.cpu().double().t()).to(
        device, torch.float32
    )
    actual = torch.ops.vllm.mx_maca_fp32_router_gemm_dispatch(activations, weight)

    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=1e-6)
    assert actual.device == weight.device
    if tokens:
        torch.testing.assert_close(
            actual.topk(8).indices.sort().values, expected.topk(8).indices.sort().values
        )


def test_rejects_invalid_inputs(device):
    MacaPlatform.import_kernels()
    weight = torch.ones(256, 3072, device=device)
    a = torch.ones(33, 3072, device=device)
    out = torch.empty(33, 256, device=device)
    with pytest.raises(RuntimeError, match="num_tokens exceeds"):
        torch.ops._C.fp32_router_gemm(out, a, weight)
    with pytest.raises(RuntimeError, match="must be contiguous"):
        torch.ops._C.fp32_router_gemm(out[:1], a[:2, :].t()[:1, :], weight)


def test_dispatch_large_batch_uses_fp32_weights(device):
    column_ids = torch.arange(33, device=device) * 89
    values = torch.arange(256 * 3072, device=device).remainder(37)
    weight = (1.0 + values.float() / 1024).view(256, 3072)
    a = torch.zeros(33, 3072, dtype=torch.bfloat16, device=device)
    a.scatter_(1, column_ids[:, None], 1.0)
    expected = weight[:, column_ids].T.contiguous()

    actual = _maca_fp32_router_gemm_dispatch(a, weight)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
