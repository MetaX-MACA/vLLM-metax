# SPDX-License-Identifier: Apache-2.0

from vllm import ModelRegistry

def register_model():
    from .baichuan import BaichuanForCausalLM
    from .baichuan_moe import BaiChuanMoEForCausalLM
    from .telechat import TelechatForCausalLM
    from .deepseek import DeepseekForCausalLM
    from .deepseek_v2 import DeepseekV2ForCausalLM
    from .qwen import QWenLMHeadModel
    from .qwen3 import Qwen3ForCausalLM
    from .qwen3_moe import Qwen3MoeForCausalLM
    from .mimo import MiMoForCausalLM
    from .mimo_mtp import MiMoMTP

    ModelRegistry.register_model(
        "BaichuanForCausalLM",
        "vllm_metax_plugin.models.baichuan:BaichuanForCausalLM")

    ModelRegistry.register_model(
        "BaiChuanMoEForCausalLM",
        "vllm_metax_plugin.models.baichuan_moe:BaiChuanMoEForCausalLM")

    ModelRegistry.register_model(
        "TelechatForCausalLM",
        "vllm_metax_plugin.models.telechat:TelechatForCausalLM")

    ModelRegistry.register_model(
        "DeepseekV2ForCausalLM",
        "vllm_metax_plugin.models.deepseek_v2:DeepseekV2ForCausalLM")

    ModelRegistry.register_model(
        "DeepseekV3ForCausalLM",
        "vllm_metax_plugin.models.deepseek_v2:DeepseekV3ForCausalLM")

    ModelRegistry.register_model(
        "MiMoForCausalLM",
        "vllm_metax_plugin.models.mimo:MiMoForCausalLM")

    ModelRegistry.register_model(
        "MiMoMTPModel",
        "vllm_metax_plugin.models.mimo_mtp:MiMoMTP")

    ModelRegistry.register_model(
        "QWenLMHeadModel",
        "vllm_metax_plugin.models.qwen:QWenLMHeadModel")

    ModelRegistry.register_model(
        "Qwen3ForCausalLM",
        "vllm_metax_plugin.models.qwen3:Qwen3ForCausalLM")
    
    ModelRegistry.register_model(
        "Qwen3MoeForCausalLM",
        "vllm_metax_plugin.models.qwen3_moe:Qwen3MoeForCausalLM")
