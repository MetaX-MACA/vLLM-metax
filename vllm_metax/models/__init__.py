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

    ModelRegistry.register_model(
        "GlmMoeDsaForCausalLM", "vllm_metax.models.deepseek_v2:GlmMoeDsaForCausalLM"
    )

    ModelRegistry.register_model(
        "DeepseekV4ForCausalLM", "vllm_metax.models.deepseek_v4:DeepseekV4ForCausalLM",
    )

    ModelRegistry.register_model(
        "DeepSeekV4MTPModel", "vllm_metax.models.deepseek_v4:DeepSeekV4MTP"
    )

    ModelRegistry.register_model(
        "Step3p5MTP", "vllm_metax.models.step3p5_mtp:Step3p5MTP"
    )

    ModelRegistry.register_model(
        "MiMoV2ForCausalLM", "vllm_metax.models.mimo_v2:MiMoV2ForCausalLM"
    )

    ModelRegistry.register_model(
        "MiMoV2FlashForCausalLM", "vllm_metax.models.mimo_v2:MiMoV2FlashForCausalLM"
    )

    ModelRegistry.register_model(
        "MiMoV2MTPModel", "vllm_metax.models.mimo_v2_mtp:MiMoV2MTP"
    )

    ModelRegistry.register_model(
        "MiMoV2OmniMTPModel", "vllm_metax.models.mimo_v2_mtp:MiMoV2OmniMTP"
    )