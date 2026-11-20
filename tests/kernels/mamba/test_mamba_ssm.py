# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch
from torch.nn import functional as F

from vllm.model_executor.layers.mamba.ops.mamba_ssm import selective_scan_fn

from vllm_metax.platform import MacaPlatform

pytestmark = pytest.mark.requires_maca


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scan_chunks", [1, 3])
@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("has_gate", [False, True])
def test_selective_scan(device, dtype, scan_chunks, groups, has_gate):
    MacaPlatform.import_kernels()
    batch, channels, dstate, length = 2, 4, 8, 129
    rng = torch.Generator(device=device).manual_seed(0)
    u = torch.randn(batch, channels, length, device=device, dtype=dtype, generator=rng)
    delta = torch.rand(u.shape, device=device, dtype=dtype, generator=rng) / 2
    a = -torch.rand(channels, dstate, device=device, generator=rng) / 2
    b = torch.randn(
        batch, groups, dstate, length, device=device, dtype=dtype, generator=rng
    )
    c = torch.randn(b.shape, device=device, dtype=dtype, generator=rng)
    d = torch.randn(channels, device=device, generator=rng)
    bias = torch.rand(channels, device=device, generator=rng)
    z = (
        torch.randn(u.shape, device=device, dtype=dtype, generator=rng)
        if has_gate
        else None
    )
    state = torch.full((batch, channels, dstate), -99, device=device, dtype=dtype)

    delta_ref = F.softplus(delta.float() + bias[:, None])
    b_ref = b.float().repeat_interleave(channels // groups, dim=1)
    c_ref = c.float().repeat_interleave(channels // groups, dim=1)
    state_ref = torch.zeros_like(state, dtype=torch.float32)
    outputs = []
    for t in range(length):
        dt = delta_ref[..., t, None]
        value = u[..., t].float()
        state_ref = (dt * a).exp() * state_ref + dt * value[..., None] * b_ref[..., t]
        outputs.append((state_ref * c_ref[..., t]).sum(-1) + d * value)
    expected = torch.stack(outputs, dim=-1)
    if z is not None:
        expected *= F.silu(z.float())

    result = []
    for chunk in range(scan_chunks):
        tokens = slice(
            chunk * length // scan_chunks, (chunk + 1) * length // scan_chunks
        )
        initial = torch.ones(batch, dtype=torch.bool, device=device) if chunk else None
        result.append(
            selective_scan_fn(
                u[..., tokens].contiguous(),
                state,
                delta[..., tokens].contiguous().clone(),
                a,
                b[..., tokens],
                c[..., tokens],
                d,
                z=None if z is None else z[..., tokens].contiguous().clone(),
                delta_bias=bias,
                delta_softplus=True,
                has_initial_state=initial,
                block_size=2048,
            )
        )

    rtol, atol = {
        torch.float32: (6e-4, 2e-3),
        torch.float16: (3e-3, 5e-3),
        torch.bfloat16: (3e-2, 5e-2),
    }[dtype]
    torch.testing.assert_close(
        torch.cat(result, dim=-1), expected.to(dtype), atol=atol, rtol=rtol
    )
    torch.testing.assert_close(state, state_ref.to(dtype), atol=atol, rtol=rtol)
