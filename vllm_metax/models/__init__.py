# SPDX-License-Identifier: Apache-2.0
# 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved. 

from vllm import ModelRegistry


def register_model():

    ModelRegistry.register_model(
        "BaichuanForCausalLM",
        "vllm_metax.models.baichuan:BaichuanForCausalLM")

    ModelRegistry.register_model(
        "BaiChuanMoEForCausalLM",
        "vllm_metax.models.baichuan_moe:BaiChuanMoEForCausalLM")

    ModelRegistry.register_model(
        "Qwen2VLForConditionalGeneration",
        "vllm_metax.models.qwen2_vl:Qwen2VLForConditionalGeneration")

    ModelRegistry.register_model(
        "InternVLChatModel",
        "vllm_metax.models.internvl:InternVLChatModel")

    ModelRegistry.register_model("DeepSeekMTPModel",
                                 "vllm_metax.models.deepseek_mtp:DeepSeekMTP")