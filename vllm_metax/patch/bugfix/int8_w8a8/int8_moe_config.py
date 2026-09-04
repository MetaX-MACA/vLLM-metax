# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# -----------------------------------------------------------------------
# Note: MiniMax M3 (and other clamped-SwiGLU models) configure per-layer
#       gemm1_alpha/gemm1_beta/gemm1_clamp_limit, but vLLM's
#       int8_w8a16_moe_quant_config()/int8_w8a8_moe_quant_config() don't
#       accept or forward these to FusedMoEQuantConfig. Without them,
#       TritonExperts.activation() asserts on SWIGLUOAI_UNINTERLEAVE
#       ("requires gemm1_clamp_limit").
#
#       This mirrors https://github.com/vllm-project/vllm/pull/47552
#       (JianDan0212:fix-minimax-m3-int8 -> vllm-project:main), which adds
#       gemm1_alpha/beta/clamp_limit params to both functions below and
#       forwards them into FusedMoEQuantConfig.
#
# Affected versions: v0.24.0 (PR #47552 not yet merged)
#
# Remove at: once PR #47552 (or equivalent) merges into a vLLM release this
#            plugin targets.
# -----------------------------------------------------------------------

import torch

from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
)
from vllm_metax.patch import patch


@patch(target_module_path="vllm.model_executor.layers.fused_moe.config")
def int8_w8a8_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    a1_scale: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    per_act_token_quant: bool = False,
    # ┌------------------------  Metax Modification -------------------------┐
    gemm1_alpha: float | None = None,
    gemm1_beta: float | None = None,
    gemm1_clamp_limit: float | None = None,
    # └------------------------- Metax Modification -------------------------┘
) -> FusedMoEQuantConfig:
    """
    Construct a quant config for int8 activations and int8 weights.
    """
    return FusedMoEQuantConfig.make(
        torch.int8,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        per_act_token_quant=per_act_token_quant,
        per_out_ch_quant=False,
        block_shape=None,
        # ┌--------------------  Metax Modification ---------------------┐
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
        # └-------------------------------------------------------------┘
    )
