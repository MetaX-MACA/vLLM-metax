# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch
from torch.nn import functional as F

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

from vllm_metax.model_executor.layers.fused_moe.fused_moe import fused_experts

pytestmark = pytest.mark.requires_maca

BLOCK = 128


def _activation_roundtrip(x):
    blocks = x.float().unflatten(-1, (-1, BLOCK))
    scale = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-10) / 127
    return ((blocks / scale).trunc().clamp(-128, 127) * scale).flatten(-2)


def _weight_roundtrip(w, scales):
    return w.float() * scales.repeat_interleave(BLOCK, -2).repeat_interleave(BLOCK, -1)


@pytest.mark.parametrize("tokens", [1, 33])
@pytest.mark.parametrize("hidden,intermediate", [(128, 128), (512, 256)])
@torch.inference_mode()
def test_w8a8_block_int8_fused_moe(device, tokens, hidden, intermediate):
    gen = torch.Generator(device=device).manual_seed(0)
    experts = 8
    dtype = torch.bfloat16
    inputs = torch.randn(tokens, hidden, device=device, dtype=dtype, generator=gen) / 10
    w1 = torch.randint(
        -127,
        128,
        (experts, 2 * intermediate, hidden),
        dtype=torch.int8,
        device=device,
        generator=gen,
    )
    w2 = torch.randint(
        -127,
        128,
        (experts, hidden, intermediate),
        dtype=torch.int8,
        device=device,
        generator=gen,
    )
    s1 = (
        torch.rand(
            experts,
            2 * intermediate // BLOCK,
            hidden // BLOCK,
            device=device,
            generator=gen,
        )
        / 360
    )
    s2 = (
        torch.rand(
            experts,
            hidden // BLOCK,
            intermediate // BLOCK,
            device=device,
            generator=gen,
        )
        / 360
    )
    seq = torch.arange(tokens, device=device, dtype=torch.int32)
    topk_ids = torch.stack((seq % experts, (seq + 3) % experts), dim=-1)
    topk_weights = (
        torch.tensor([0.75, 0.25], device=device).expand(tokens, 2).contiguous()
    )
    config = FusedMoEQuantConfig.make(
        torch.int8,
        w1_scale=s1,
        w2_scale=s2,
        block_shape=[BLOCK, BLOCK],
    )

    init_workspace_manager(device)
    try:
        with set_current_vllm_config(VllmConfig()):
            actual = fused_experts(
                inputs, w1, w2, topk_weights, topk_ids, quant_config=config
            )
    finally:
        reset_workspace_manager()

    expected = torch.zeros_like(inputs, dtype=torch.float32)
    dequant_x = _activation_roundtrip(inputs)
    for expert in range(experts):
        gate, up = (
            F.linear(dequant_x, _weight_roundtrip(w1[expert], s1[expert]))
            .to(dtype)
            .chunk(2, dim=-1)
        )
        activated = (F.silu(gate.float()) * up.float()).to(dtype)
        expert_out = F.linear(
            _activation_roundtrip(activated), _weight_roundtrip(w2[expert], s2[expert])
        ).to(dtype)
        weight = (topk_weights * (topk_ids == expert)).sum(-1, keepdim=True)
        expected += expert_out.float() * weight
    torch.testing.assert_close(actual, expected.to(dtype), atol=2e-3, rtol=6.5e-2)
