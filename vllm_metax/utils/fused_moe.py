# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from vllm_metax.model_executor.layers.fused_moe.fused_moe import (
    TritonExperts as mx_TritonExperts,
    fused_experts as vllm_fused_experts,
)

from vllm.model_executor.layers.fused_moe.fused_moe import (
    TritonExperts as vllm_TritonExperts,
    fused_experts as mx_fused_experts,
    logger,
)


import vllm_metax.envs as mx_envs


def get_triton_experts_cls():
    if mx_envs.USE_VLLM_TRITON_EXPERT:
        logger.info(
            "Using vLLM's fused MoE implementation for debugging and comparison."
        )
        return vllm_TritonExperts
    return mx_TritonExperts


def get_fused_experts_fn():
    if mx_envs.USE_VLLM_TRITON_EXPERT:
        logger.info(
            "Using vLLM's fused MoE implementation for debugging and comparison."
        )
        return vllm_fused_experts
    return mx_fused_experts
