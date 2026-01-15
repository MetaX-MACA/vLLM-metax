# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

from vllm import ModelRegistry


def register_model():

    ModelRegistry.register_model(
        "Qwen2VLForConditionalGeneration",
        "vllm_metax.models.qwen2_vl:Qwen2VLForConditionalGeneration",
    )

    ModelRegistry.register_model(
        "Qwen2_5_VLForConditionalGeneration",
        "vllm_metax.models.qwen2_5_vl:Qwen2_5_VLForConditionalGeneration",
    )

    ModelRegistry.register_model(
        "Qwen3VLForConditionalGeneration",
        "vllm_metax.models.qwen3_vl:Qwen3VLForConditionalGeneration",
    )

    #ModelRegistry.register_model(
    #    "InternVLChatModel",
    #    "vllm_metax.models.internvl:InternVLChatModel")

    # ModelRegistry.register_model(
    #     "DeepSeekMTPModel", "vllm_metax.models.deepseek_mtp:DeepSeekMTP"
    # )

    # ModelRegistry.register_model(
    #     "DeepseekV2ForCausalLM", "vllm_metax.models.deepseek_v2:DeepseekV2ForCausalLM"
    # )

    # ModelRegistry.register_model(
    #     "DeepseekV3ForCausalLM", "vllm_metax.models.deepseek_v2:DeepseekV3ForCausalLM"
    # )

    ModelRegistry.register_model(
        "DeepseekV32ForCausalLM", "vllm_metax.models.deepseek_v2:DeepseekV3ForCausalLM"
    )
