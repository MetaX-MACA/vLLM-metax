# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

import vllm.model_executor.models.registry

# Patch the transformers_utils.configs module
import vllm_metax.patch.transformers_utils.configs


# Patch the _MULTIMODAL_MODELS dictionary to add Qwen3ASRForConditionalGeneration
def patch_multimodal_models():
    # Get the existing _MULTIMODAL_MODELS dictionary
    multimodal_models = vllm.model_executor.models.registry._MULTIMODAL_MODELS

    # Add the Qwen3ASRForConditionalGeneration model
    if "Qwen3ASRForConditionalGeneration" not in multimodal_models:
        multimodal_models["Qwen3ASRForConditionalGeneration"] = (
            "qwen3_asr",
            "Qwen3ASRForConditionalGeneration",
        )
        print(
            "Successfully added Qwen3ASRForConditionalGeneration to _MULTIMODAL_MODELS"
        )
    else:
        print("Qwen3ASRForConditionalGeneration is already in _MULTIMODAL_MODELS")


# Apply the patch
patch_multimodal_models()
