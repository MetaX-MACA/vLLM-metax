# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# NOTE(metax): Start from the NVIDIA model; AMD's cache and kernel contracts differ.
from vllm_metax.models.minimax_m3.metax.dspark import MiniMaxM3DSparkForCausalLM
from vllm_metax.models.minimax_m3.metax.model import (
    MiniMaxM3SparseForCausalLM,
    MiniMaxM3SparseForConditionalGeneration,
)
from vllm_metax.models.minimax_m3.metax.mtp import MiniMaxM3MTP

__all__ = [
    "MiniMaxM3DSparkForCausalLM",
    "MiniMaxM3MTP",
    "MiniMaxM3SparseForCausalLM",
    "MiniMaxM3SparseForConditionalGeneration",
]
