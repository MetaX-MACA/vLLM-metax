# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import vllm.transformers_utils.configs as configs_module
import vllm.transformers_utils.config as config_registry_module
from transformers import AutoConfig
from vllm.logger import init_logger

# [CRITICAL]
from .qwen3_asr import Qwen3ASRConfig

logger = init_logger(__name__)


def patch_all_registries():
    # ------------------------------------------------------------------
    # tell vLLM that Qwen3ASRConfig is in qwen3_asr.py
    # ------------------------------------------------------------------
    if "Qwen3ASRConfig" not in configs_module.__all__:
        configs_module.__all__.append("Qwen3ASRConfig")

    target_dict_name = "_CONFIG_REGISTRY"
    if not hasattr(configs_module, target_dict_name) and hasattr(
        configs_module, "_CLASS_TO_MODULE"
    ):
        target_dict_name = "_CLASS_TO_MODULE"

    if hasattr(configs_module, target_dict_name):
        file_registry = getattr(configs_module, target_dict_name)
        file_registry["Qwen3ASRConfig"] = (
            "vllm_metax.patch.transformers_utils.configs.qwen3_asr"
        )
        logger.info(f"Updated vLLM file registry {target_dict_name}")

    # ------------------------------------------------------------------
    # tell vLLM that Qwen3ASRConfig is qwen3_asr
    # ------------------------------------------------------------------
    try:
        model_type_registry = config_registry_module._CONFIG_REGISTRY
        if "qwen3_asr" not in model_type_registry:
            model_type_registry["qwen3_asr"] = "Qwen3ASRConfig"
            logger.info("Updated vLLM internal model_type registry")
    except Exception as e:
        logger.error(f"Failed to patch vLLM internal registry: {e}")


# Apply the patch
patch_all_registries()
