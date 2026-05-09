# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

from vllm import ModelRegistry


def register_model():

    ModelRegistry.register_model(
        "DeepSeekMTPModel", "vllm_metax.models.deepseek_mtp:DeepSeekMTP"
    )

    ModelRegistry.register_model(
        "DeepseekV2ForCausalLM", "vllm_metax.models.deepseek_v2:DeepseekV2ForCausalLM"
    )

    ModelRegistry.register_model(
        "DeepseekV3ForCausalLM", "vllm_metax.models.deepseek_v2:DeepseekV3ForCausalLM"
    )

    ModelRegistry.register_model(
        "DeepseekV32ForCausalLM", "vllm_metax.models.deepseek_v2:DeepseekV3ForCausalLM"
    )

    # TODO(hank): we may not need to manually register it since the metax dependency is removed.
    ModelRegistry.register_model(
        "KimiK25ForConditionalGeneration", "vllm_metax.models.kimi_k25:KimiK25ForConditionalGeneration"
    )

    ModelRegistry.register_model(
        "GlmMoeDsaForCausalLM", "vllm_metax.models.deepseek_v2:GlmMoeDsaForCausalLM"
    )

    ModelRegistry.register_model(
        "DeepseekV4ForCausalLM", "vllm_metax.models.deepseek_v4:DeepseekV4ForCausalLM",
    )

    ModelRegistry.register_model(
        "DeepSeekV4MTPModel", "vllm_metax.models.deepseek_v4_mtp:DeepSeekV4MTP"
    )

    ModelRegistry.register_model(
        "MiniMaxM2ForCausalLM", "vllm_metax.models.minimax_m2:MiniMaxM2ForCausalLM",
    )

    ModelRegistry.register_model(
        "DeepSeekV4MTPModel", "vllm_metax.models.deepseek_v4_mtp:DeepSeekV4MTP"
    )