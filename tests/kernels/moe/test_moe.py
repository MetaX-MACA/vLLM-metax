# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch
import torch.nn.functional as F

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

from vllm_metax.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm_metax.platform import MacaPlatform


@pytest.fixture
def moe_workspace(device):
    init_workspace_manager(device)
    try:
        with set_current_vllm_config(VllmConfig()):
            yield
    finally:
        reset_workspace_manager()


# Adapted from vLLM v0.33.0 tests/kernels/moe/test_moe.py.
@pytest.mark.requires_maca
@pytest.mark.parametrize("tokens", [1, 17])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("with_remote_experts", [False, True])
def test_metax_fused_experts_matches_pytorch(
    device,
    moe_workspace,
    tokens,
    inplace,
    with_remote_experts,
):
    torch.manual_seed(7)
    dtype = torch.bfloat16
    experts, hidden, intermediate = 4, 128, 128
    x = torch.randn(tokens, hidden, device=device, dtype=dtype) / 10
    w1 = torch.randn(experts, 2 * intermediate, hidden, device=device, dtype=dtype) / 10
    w2 = torch.randn(experts, hidden, intermediate, device=device, dtype=dtype) / 10
    token_indices = torch.arange(tokens, device=device, dtype=torch.int32)
    topk_ids = torch.stack(
        (token_indices % experts, (token_indices + 1) % experts), dim=1
    )
    topk_weights = torch.tensor([[0.75, 0.25]], device=device).repeat(tokens, 1)
    expert_map = None
    expert_indices = [0, 1, 2, 3]
    if with_remote_experts:
        expert_indices = [1, 3]
        expert_map = torch.tensor([-1, 0, -1, 1], device=device, dtype=torch.int32)

    actual_input = x.clone()
    actual = fused_experts(
        actual_input,
        w1[expert_indices],
        w2[expert_indices],
        topk_weights,
        topk_ids,
        inplace=inplace,
        global_num_experts=experts,
        expert_map=expert_map,
    )
    if inplace:
        assert actual.data_ptr() == actual_input.data_ptr()
    else:
        torch.testing.assert_close(actual_input, x, atol=0, rtol=0)

    reference = torch.zeros_like(x).float()
    for expert in expert_indices:
        gate, up = F.linear(x, w1[expert]).chunk(2, dim=-1)
        contribution = F.linear(F.silu(gate) * up, w2[expert])
        weight = (topk_weights * (topk_ids == expert)).sum(dim=-1, keepdim=True)
        reference += contribution.float() * weight
    torch.testing.assert_close(actual, reference.to(dtype), atol=1e-3, rtol=1.6e-2)


# Adapted from vLLM v0.33.0 tests/kernels/moe/test_moe.py.
@pytest.mark.requires_maca
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("topk", [2, 8])
@pytest.mark.parametrize("strided", [False, True])
def test_moe_sum_ignores_unwritten_routes(device, dtype, topk, strided):
    MacaPlatform.import_kernels()
    torch.manual_seed(0)
    values = torch.randn(
        17, topk * (2 if strided else 1), 131, device=device, dtype=dtype
    )
    if strided:
        values = values[:, ::2]
    expert_map = torch.tensor([0, -1, 1, -1], device=device, dtype=torch.int32)
    ids = torch.randint(
        -1, len(expert_map), (17, topk), device=device, dtype=torch.int32
    )
    ids[0].fill_(-1)
    valid = (ids >= 0) & (expert_map[ids.clamp_min(0).long()] >= 0)
    values.masked_fill_(~valid[..., None], torch.nan)
    expected = values.masked_fill(~valid[..., None], 0).float().sum(1).to(dtype)
    actual = torch.full_like(expected, torch.nan)
    torch.ops._moe_C.moe_sum(values, actual, ids, expert_map)
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=0)
    assert (actual[0] == 0).all()
