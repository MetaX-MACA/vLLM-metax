# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
    MODEL_ARCH_CONFIG_CONVERTORS,
)


class JoyAIModelArchConfigConvertor(ModelArchConfigConvertorBase):
    def is_deepseek_mla(self) -> bool:
        if not hasattr(self.hf_text_config, "model_type"):
            return False
        elif self.hf_text_config.model_type in (
            "AXK1",
            "deepseek_v2",
            "deepseek_v3",
            "deepseek_v32",
            "deepseek_v4",
            "deepseek_mtp",
            # /------------------------ metax modified ------------------------\ #
            "joyai_llm_flash",  # # JoyAI_LLM_Flash
            # \----------------------------------------------------------------/ #
            "glm_moe_dsa",
            "glm4_moe_lite",
            "glm4_moe_lite_mtp",
            "kimi_k2",
            "kimi_linear",
            "longcat_flash",
            "pangu_ultra_moe",
            "pangu_ultra_moe_mtp",
            "bailing_hybrid",
        ):
            # check is deepseek_v4 model
            if hasattr(self.hf_text_config, "compress_ratios"):
                return getattr(self.hf_text_config, "head_dim", None) is not None
            else:
                return getattr(self.hf_text_config, "kv_lora_rank", None) is not None
        elif self.hf_text_config.model_type == "eagle":
            # if the model is an EAGLE module, check for the
            # underlying architecture
            return (
                self.hf_text_config.model.model_type
                in (
                    "AXK1",
                    "deepseek_v2",
                    "deepseek_v3",
                    "deepseek_v32",
                    "deepseek_mtp",
                )
                and getattr(self.hf_text_config, "kv_lora_rank", None) is not None
            )
        return False


MODEL_ARCH_CONFIG_CONVERTORS["joyai_llm_flash"] = JoyAIModelArchConfigConvertor
