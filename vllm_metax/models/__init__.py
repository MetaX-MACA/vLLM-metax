# SPDX-License-Identifier: Apache-2.0

from vllm import ModelRegistry


def register_model():
    # ModelRegistry.register_model(
    #     "BaichuanForCausalLM", "vllm_metax.models.baichuan:BaichuanForCausalLM"
    # )

    # ModelRegistry.register_model(
    #     "BaiChuanMoEForCausalLM",
    #     "vllm_metax.models.baichuan_moe:BaiChuanMoEForCausalLM",
    # )

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

    # ModelRegistry.register_model(
    #     "DeepSeekMTPModel", "vllm_metax.models.deepseek_mtp:DeepSeekMTP"
    # )

    ModelRegistry.register_model(
        "DeepseekV2ForCausalLM", "vllm_metax.models.deepseek_v2:DeepseekV2ForCausalLM"
    )

    ModelRegistry.register_model(
        "DeepseekV3ForCausalLM", "vllm_metax.models.deepseek_v2:DeepseekV3ForCausalLM"
    )

    ModelRegistry.register_model(
        "DeepseekV32ForCausalLM", "vllm_metax.models.deepseek_v2:DeepseekV3ForCausalLM"
    )
