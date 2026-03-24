# SPDX-License-Identifier: Apache-2.0
from vllm.transformers_utils.config import LazyConfigDict
from vllm.transformers_utils.config import _CONFIG_REGISTRY
from vllm.transformers_utils.configs import _CLASS_TO_MODULE
import vllm.transformers_utils.config as cfg

_CONFIG_REGISTRY.update(
    LazyConfigDict(qwen3_5="Qwen3_5Config", qwen3_5_moe="Qwen3_5MoeConfig,")
)
_CLASS_TO_MODULE.update(
    {
        "Qwen3_5Config": "vllm_metax.patch.transformers_utils.configs.qwen3_5",
        "Qwen3_5TextConfig": "vllm_metax.patch.transformers_utils.configs.qwen3_5",
        "Qwen3_5MoeConfig": "vllm_metax.patch.transformers_utils.configs.qwen3_5_moe",
        "Qwen3_5MoeTextConfig": "vllm_metax.patch.transformers_utils.configs.qwen3_5_moe",
    }
)
