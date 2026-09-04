# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from vllm.transformers_utils.model_arch_config_convertor import (
    ModelArchConfigConvertorBase,
    MODEL_ARCH_CONFIG_CONVERTORS,
)


# Just for JoyAI LLM Flash model, which is a modified version of DeepSeek V3 model.
class JoyAIModelArchConfigConvertor(ModelArchConfigConvertorBase):
    def is_deepseek_mla(self) -> bool:
        if not hasattr(self.hf_text_config, "model_type"):
            return False
        elif self.hf_text_config.model_type in (
            # /------------------------ metax modified ------------------------\ #
            "joyai_llm_flash",
            # \----------------------------------------------------------------/ #
        ):
            # check is deepseek_v4 model
            if hasattr(self.hf_text_config, "compress_ratios"):
                return getattr(self.hf_text_config, "head_dim", None) is not None
            else:
                return getattr(self.hf_text_config, "kv_lora_rank", None) is not None

        return False


MODEL_ARCH_CONFIG_CONVERTORS["joyai_llm_flash"] = JoyAIModelArchConfigConvertor
