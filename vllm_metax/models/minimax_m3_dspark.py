# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Compatibility export for the former standalone MiniMax DSpark module."""

from vllm_metax.models.minimax_m3.metax.dspark import (  # noqa: F401
    MiniMaxM3DSparkDecoderLayer,
    MiniMaxM3DSparkForCausalLM,
    MiniMaxM3DSparkMLP,
    MiniMaxM3DSparkModel,
    map_draft_quant_config,
    rename_checkpoint_weights,
)
