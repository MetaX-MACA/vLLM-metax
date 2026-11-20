# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch

from vllm_metax.model_executor.layers.online_rotation import CompositeHadamardRotation

pytestmark = pytest.mark.requires_maca


def _sylvester_matrix(size):
    matrix = torch.ones(1, 1, dtype=torch.float32)
    while matrix.size(0) < size:
        matrix = torch.cat(
            (torch.cat((matrix, matrix), dim=1), torch.cat((matrix, -matrix), dim=1)),
            dim=0,
        )
    return matrix


@pytest.mark.parametrize(
    "hidden_size",
    [
        pytest.param(
            8,
            marks=pytest.mark.xfail(
                raises=ValueError,
                strict=True,
                reason="TileLang Hadamard has no autotune configurations for P < 16",
            ),
        ),
        32,
    ],
)
@torch.inference_mode()
def test_hadacore(device, hidden_size):
    rng = torch.Generator(device=device).manual_seed(23)
    inputs = torch.randn(
        2, 3, hidden_size, device=device, dtype=torch.bfloat16, generator=rng
    )
    indices = torch.arange(hidden_size, device=device)
    sign = torch.where(indices % 3 == 1, -1, 1).to(torch.int8)
    rotation = CompositeHadamardRotation(
        hidden_size,
        sign,
        torch.ones(1, 1, device=device),
        k=1,
        p=hidden_size,
    )
    matrix = _sylvester_matrix(hidden_size).to(device) * hidden_size**-0.5
    expected = ((inputs.float() * sign) @ matrix).to(inputs.dtype)

    output = rotation(inputs)
    restored = rotation(output, inverse=True)

    torch.testing.assert_close(output, expected, atol=2e-2, rtol=1.6e-2)
    torch.testing.assert_close(restored, inputs, atol=2e-2, rtol=1.6e-2)
