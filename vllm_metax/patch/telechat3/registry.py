# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

import importlib.util
import sys

spec = importlib.util.find_spec("vllm.model_executor.models.registry")
if spec is None or spec.origin is None:
    raise RuntimeError("vLLM registry not found")

registry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(registry)

MODEL_MAP_PATCH = {
    "Telechat3ForCausalLM": ("llama", "LlamaForCausalLM"),
    "TeleChat3ForCausalLM": ("llama", "LlamaForCausalLM"),
}
registry._TEXT_GENERATION_MODELS.update(MODEL_MAP_PATCH)
registry._EMBEDDING_MODELS.update(registry._TEXT_GENERATION_MODELS)
registry._VLLM_MODELS.update(registry._EMBEDDING_MODELS)

from vllm.model_executor.models import registry

registry.ModelRegistry.models.pop("Telechat3ForCausalLM", None)
registry.ModelRegistry.register_model(
    "Telechat3ForCausalLM", "vllm.model_executor.models.llama:LlamaForCausalLM"
)

sys.modules["vllm.model_executor.models.registry"] = registry
