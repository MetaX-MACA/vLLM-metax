# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def test_custom_op_schemas_allow_act_quant_fusion_import():
    import torch

    import vllm_metax.compat  # noqa: F401

    assert hasattr(torch.ops._C.scaled_fp4_quant, "out")
    assert hasattr(torch.ops._C, "silu_and_mul_per_block_quant")
    if hasattr(torch, "accelerator"):
        assert hasattr(torch.accelerator, "empty_cache")

    import vllm.compilation.passes.fusion.act_quant_fusion as act_quant_fusion

    assert act_quant_fusion.SILU_MUL_OP is not None
