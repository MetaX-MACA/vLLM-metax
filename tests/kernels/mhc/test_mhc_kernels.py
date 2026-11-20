# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.mhc import MHCPostOp as VllmMHCPostOp

from vllm_metax.registry.custom_ops.layers.mhc import MHCPostOp


@pytest.mark.requires_maca
@pytest.mark.parametrize(
    ("num_tokens", "hc_mult", "hidden_size"),
    [(n, 4, d) for n in (1, 4, 8, 128) for d in (4096, 7168)]
    + [(3, 2, 256), (33, 2, 256), (3, 4, 512)],
)
def test_mhc_post_tilelang(device, num_tokens, hc_mult, hidden_size):
    rng = torch.Generator(device=device).manual_seed(0)
    x = torch.randn(
        num_tokens, hidden_size, device=device, dtype=torch.bfloat16, generator=rng
    )
    residual = torch.randn(
        num_tokens, hc_mult, hidden_size, device=device, dtype=x.dtype, generator=rng
    )
    post_mix = torch.randn(num_tokens, hc_mult, 1, device=device, generator=rng)
    comb_mix = torch.randn(num_tokens, hc_mult, hc_mult, device=device, generator=rng)
    expected = (
        x.float().unsqueeze(-2) * post_mix + torch.bmm(comb_mix.mT, residual.float())
    ).to(x.dtype)

    with set_current_vllm_config(VllmConfig()):
        op = VllmMHCPostOp(enforce_enable=True)
        assert type(op) is MHCPostOp
        actual = op(x, residual, post_mix, comb_mix)

    torch.testing.assert_close(actual, expected, atol=5e-2, rtol=1e-2)
