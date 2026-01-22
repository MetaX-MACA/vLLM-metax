# SPDX-License-Identifier: Apache-2.0

from vllm import ModelRegistry


def register_model():

    # support dsv32
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
