# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

from vllm import ModelRegistry


def register_model():
    ModelRegistry.register_model(
        "DeepSeekMTPModel", "vllm_metax.models.deepseek_mtp:DeepSeekMTP"
    )

    # NOTE(MetaX): vLLM 0.29 resolves the V3.2 style MTP draft
    # (deepseek_v32 / glm_moe_dsa, e.g. GLM-5.2) to
    # `vllm.models.deepseek_v32:DeepseekV32MTP`, which is the upstream
    # DeepSeekMTP implementation. That implementation always quantizes the
    # sparse attention indexer Q to FP8, which the MetaX sparse indexer op only
    # accepts on FP8 capable platforms (VLLM_METAX_SUPPORTS_FP8). Keep the
    # MetaX subclass here so the MTP draft uses the MetaX sparse indexer and
    # indexer cache implementation, matching the MetaX V3.2 target model.
    # Revisit when the upstream generic decoder uses the MetaX kernels directly.
    ModelRegistry.register_model(
        "DeepseekV32MTPModel", "vllm_metax.models.deepseek_mtp:DeepSeekMTP"
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
        "DeepseekV4ForCausalLM",
        "vllm_metax.models.deepseek_v4:DeepseekV4ForCausalLM",
    )

    ModelRegistry.register_model(
        "DeepSeekV4MTPModel", "vllm_metax.models.deepseek_v4:DeepSeekV4MTP"
    )

    ModelRegistry.register_model(
        "DSparkDraftModel", "vllm_metax.models.deepseek_v4:DSparkDeepseekV4ForCausalLM"
    )

    ModelRegistry.register_model(
        "Step3p5MTP", "vllm_metax.models.step3p5_mtp:Step3p5MTP"
    )

    ModelRegistry.register_model(
        "Step3p5ForCausalLM", "vllm_metax.models.step3p5:Step3p5ForCausalLM"
    )

    ModelRegistry.register_model(
        "Qwen4ExpForCausalLM", "vllm_metax.models.qwen4_exp:Qwen4ExpForCausalLM"
    )

    ModelRegistry.register_model(
        "Qwen4ExpForConditionalGeneration",
        "vllm_metax.models.qwen4_exp:Qwen4ExpForConditionalGeneration",
    )

    ModelRegistry.register_model(
        "Qwen4ExpMTP", "vllm_metax.models.qwen4_exp:Qwen4ExpMTP"
    )
