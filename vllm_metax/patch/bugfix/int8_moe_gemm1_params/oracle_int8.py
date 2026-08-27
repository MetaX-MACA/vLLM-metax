# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# -----------------------------------------------------------------------
# Note: Companion patch to fused_moe_config.py in this package. vLLM's
#       oracle/int8.py imported int8_w8a16_moe_quant_config/
#       int8_w8a8_moe_quant_config via `from ...config import ...`, which
#       binds its OWN local names at import time -- patching config.py's
#       module attribute alone does not change those already-bound names,
#       so make_int8_moe_quant_config() here must be replaced too,
#       explicitly calling the patched versions so gemm1_alpha/beta/
#       clamp_limit reach FusedMoEQuantConfig.
#
#       Mirrors https://github.com/vllm-project/vllm/pull/47552
#       (JianDan0212:fix-minimax-m3-int8 -> vllm-project:main).
#
# Affected versions: v0.24.0 (PR #47552 not yet merged)
#
# Remove at: once PR #47552 (or equivalent) merges into a vLLM release this
#            plugin targets.
# -----------------------------------------------------------------------

import torch

import vllm.model_executor.layers.fused_moe.oracle.int8 as vllm_int8
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig

# Import the already-patched versions from the sibling module (this package's
# __init__ loads fused_moe_config before oracle_int8, see __init__.py).
from .fused_moe_config import (
    int8_w8a16_moe_quant_config,
    int8_w8a8_moe_quant_config,
)


def make_int8_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    per_act_token_quant: bool = False,
    # ┌------------------------  Metax Modification -------------------------┐
    gemm1_alpha: float | None = None,
    gemm1_beta: float | None = None,
    gemm1_clamp_limit: float | None = None,
    # └------------------------- Metax Modification -------------------------┘
) -> FusedMoEQuantConfig:
    assert (a1_scale is None and a2_scale is None) or (
        a1_scale is not None and a2_scale is not None
    ), "a1_scale and a2_scale must both be provided or both be None"

    if a1_scale is None or a2_scale is None:
        return int8_w8a16_moe_quant_config(
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_zp=None,
            w2_zp=None,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            # ┌--------------------  Metax Modification ---------------------┐
            gemm1_alpha=gemm1_alpha,
            gemm1_beta=gemm1_beta,
            gemm1_clamp_limit=gemm1_clamp_limit,
            # └-------------------------------------------------------------┘
        )

    return int8_w8a8_moe_quant_config(
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        per_act_token_quant=per_act_token_quant,
        # ┌--------------------  Metax Modification ---------------------┐
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
        # └-------------------------------------------------------------┘
    )


vllm_int8.make_int8_moe_quant_config = make_int8_moe_quant_config
