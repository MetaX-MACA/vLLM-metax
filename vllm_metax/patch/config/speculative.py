from typing import TYPE_CHECKING, Any

from vllm.config.speculative import SpeculativeConfig

if TYPE_CHECKING:
    from transformers import PretrainedConfig
else:
    PretrainedConfig = Any

class MacaSpeculativeConfig(SpeculativeConfig):
    @staticmethod
    def hf_config_override(hf_config: PretrainedConfig) -> PretrainedConfig:
        initial_architecture = hf_config.architectures[0]
        if hf_config.model_type in (
            "deepseek_v3",
            "deepseek_v32",
            # /------------------------ metax modified ------------------------\ #
            "joyai_llm_flash", # JoyAI_LLM_Flash
            # \----------------------------------------------------------------/ #
            "glm_moe_dsa",
        ):
            hf_config.model_type = "deepseek_mtp"
        if hf_config.model_type == "deepseek_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["DeepSeekMTPModel"]}
            )
        if hf_config.model_type == "deepseek_v4":
            hf_config.model_type = "deepseek_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["DeepSeekV4MTPModel"]}
            )
        if hf_config.model_type in ("pangu_ultra_moe"):
            hf_config.model_type = "pangu_ultra_moe_mtp"
        if hf_config.model_type == "pangu_ultra_moe_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["OpenPanguMTPModel"]}
            )

        if hf_config.architectures[0] == "MiMoForCausalLM":
            hf_config.model_type = "mimo_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "num_hidden_layers": 0,
                    "n_predict": n_predict,
                    "architectures": ["MiMoMTPModel"],
                }
            )

        if (arch := hf_config.architectures[0]) in (
            "MiMoV2ForCausalLM",
            "MiMoV2OmniForCausalLM",
        ):
            from vllm.model_executor.models.mimo_v2_mtp import (
                _MIMO_V2_PRO_NUM_MTP_LAYERS,
            )

            mtp_arch_maps = {
                "MiMoV2ForCausalLM": "MiMoV2MTPModel",
                "MiMoV2OmniForCausalLM": "MiMoV2OmniMTPModel",
            }

            hf_config.model_type = "mimo_v2_mtp"
            # vLLM currently supports only the first MiMo-V2 MTP layer.
            n_predict = _MIMO_V2_PRO_NUM_MTP_LAYERS
            hf_config.update(
                {
                    "num_hidden_layers": 0,
                    "n_predict": n_predict,
                    "num_nextn_predict_layers": n_predict,
                    "architectures": [mtp_arch_maps[arch]],
                }
            )

        if hf_config.architectures[0] == "MiMoV2FlashForCausalLM":
            from vllm.model_executor.models.mimo_v2_mtp import (
                _MIMO_V2_FLASH_NUM_MTP_LAYERS,
            )

            hf_config.model_type = "mimo_v2_mtp"
            # vLLM currently supports only the first MiMo-V2 MTP layer.
            n_predict = _MIMO_V2_FLASH_NUM_MTP_LAYERS
            hf_config.update(
                {
                    "num_hidden_layers": 0,
                    "n_predict": n_predict,
                    "num_nextn_predict_layers": n_predict,
                    "architectures": ["MiMoV2MTPModel"],
                }
            )

        if hf_config.architectures[0] == "Glm4MoeForCausalLM":
            hf_config.model_type = "glm4_moe_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "n_predict": n_predict,
                    "architectures": ["Glm4MoeMTPModel"],
                }
            )

        if hf_config.architectures[0] == "Glm4MoeLiteForCausalLM":
            hf_config.model_type = "glm4_moe_lite_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "num_hidden_layers": 0,
                    "n_predict": n_predict,
                    "architectures": ["Glm4MoeLiteMTPModel"],
                }
            )

        if hf_config.architectures[0] == "GlmOcrForConditionalGeneration":
            hf_config.model_type = "glm_ocr_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {
                    "num_hidden_layers": 0,
                    "n_predict": n_predict,
                    "architectures": ["GlmOcrMTPModel"],
                }
            )

        if hf_config.model_type == "ernie4_5_moe":
            hf_config.model_type = "ernie_mtp"
        if hf_config.model_type == "ernie_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["ErnieMTPModel"]}
            )

        if hf_config.architectures[0] == "NemotronH_Super_Omni_Reasoning_V3":
            # Promote VLM's text_config so MTP detection below fires correctly
            hf_config = hf_config.text_config

        if (
            hf_config.model_type in {"nemotron_h", "nemotron_h_puzzle"}
            and hasattr(hf_config, "num_nextn_predict_layers")
            and hf_config.num_nextn_predict_layers > 0
        ):
            # Check if this is an MTP variant
            hf_config.model_type = "nemotron_h_mtp"
        if hf_config.model_type == "nemotron_h_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["NemotronHMTPModel"]}
            )

        if hf_config.model_type == "qwen3_next":
            hf_config.model_type = "qwen3_next_mtp"
        if hf_config.model_type == "qwen3_next_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["Qwen3NextMTP"]}
            )

        if hf_config.model_type == "exaone_moe":
            hf_config.model_type = "exaone_moe_mtp"
        if hf_config.model_type == "exaone_moe_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["ExaoneMoeMTP"]}
            )
        if "exaone4_5" in hf_config.model_type:
            hf_config.model_type = "exaone4_5_mtp"
        if hf_config.model_type == "exaone4_5_mtp":
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["Exaone4_5_MTP"]}
            )
        if hf_config.model_type in ("qwen3_5", "qwen3_5_moe"):
            is_moe = hf_config.model_type == "qwen3_5_moe"
            hf_config.model_type = "qwen3_5_mtp"
            n_predict = getattr(hf_config, "mtp_num_hidden_layers", None)
            hf_config.update(
                {
                    "n_predict": n_predict,
                    "architectures": ["Qwen3_5MoeMTP" if is_moe else "Qwen3_5MTP"],
                }
            )
        if hf_config.model_type == "intern_s2_preview":
            text_config = getattr(hf_config, "text_config", None)
            is_moe = getattr(text_config, "model_type", None) == "qwen3_5_moe_text"
            hf_config.model_type = "qwen3_5_mtp"
            n_predict = getattr(text_config, "mtp_num_hidden_layers", None)
            hf_config.update(
                {
                    "n_predict": n_predict,
                    "architectures": ["Qwen3_5MoeMTP" if is_moe else "Qwen3_5MTP"],
                }
            )
        if hf_config.model_type == "longcat_flash":
            hf_config.model_type = "longcat_flash_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["LongCatFlashMTPModel"]}
            )

        if hf_config.model_type in ("step3p5", "step3p7") or hf_config.architectures[
            0
        ] in ("Step3p5ForCausalLM", "Step3p7ForConditionalGeneration"):
            quantization_config = getattr(hf_config, "quantization_config", None)
            hf_config = getattr(hf_config, "text_config", hf_config)
            if (
                quantization_config is not None
                and getattr(hf_config, "quantization_config", None) is None
            ):
                hf_config.update({"quantization_config": quantization_config})
            hf_config.model_type = "step3p5_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", 1)
            hf_config.update({"n_predict": n_predict, "architectures": ["Step3p5MTP"]})

        if initial_architecture == "MistralLarge3ForCausalLM":
            hf_config.update({"architectures": ["EagleMistralLarge3ForCausalLM"]})

        if hf_config.model_type == "hy_v3":
            hf_config.model_type = "hy_v3_mtp"
            n_predict = getattr(hf_config, "num_nextn_predict_layers", None)
            hf_config.update(
                {"n_predict": n_predict, "architectures": ["HYV3MTPModel"]}
            )

        if hf_config.model_type in ("gemma4_assistant", "gemma4_unified_assistant"):
            hf_config.model_type = "gemma4_mtp"
            text_config = getattr(hf_config, "text_config", hf_config)
            # The assistant runs all decoder layers in a single forward
            # call to produce one draft token, so n_predict=1.
            # num_kv_shared_layers must be 0: cross-model KV sharing is
            # set up by the proposer after model construction.
            if hasattr(text_config, "num_kv_shared_layers"):
                text_config.num_kv_shared_layers = 0
            hf_config.update({"n_predict": 1, "architectures": ["Gemma4MTPModel"]})

        return hf_config


SpeculativeConfig.hf_config_override = MacaSpeculativeConfig.hf_config_override