# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# NOTE(metax): Start from the NVIDIA model; AMD's cache and kernel contracts differ.
from .metax.dspark import MiniMaxM3DSparkForCausalLM
from .metax.model import (
    MiniMaxM3SparseForCausalLM,
    MiniMaxM3SparseForConditionalGeneration,
)
from .metax.mtp import MiniMaxM3MTP

__all__ = [
    "MiniMaxM3DSparkForCausalLM",
    "MiniMaxM3MTP",
    "MiniMaxM3SparseForCausalLM",
    "MiniMaxM3SparseForConditionalGeneration",
]
