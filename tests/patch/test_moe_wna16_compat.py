# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from vllm.model_executor.layers.fused_moe.activation import MoEActivation

import vllm_metax.quant_config.moe_wna16 as moe_wna16


def test_moe_wna16_apply_tolerates_missing_disable_inplace(monkeypatch):
    captured = {}

    def fake_fused_experts(*args, **kwargs):
        captured.update(kwargs)
        return torch.empty(1)

    monkeypatch.setattr(
        moe_wna16, "get_fused_experts_fn", lambda: fake_fused_experts
    )

    method = object.__new__(moe_wna16.MoeWNA16Method)
    method.moe = SimpleNamespace()
    method.moe_quant_config = object()

    layer = SimpleNamespace(
        activation=MoEActivation.SILU,
        w13_qweight=torch.empty(1),
        w2_qweight=torch.empty(1),
        apply_router_weight_on_input=False,
        global_num_experts=1,
        expert_map=None,
    )

    result = method.apply(
        layer,
        torch.empty(1),
        torch.empty(1),
        torch.empty(1, dtype=torch.int64),
        None,
        None,
    )

    assert result.shape == (1,)
    assert captured["inplace"] is True
