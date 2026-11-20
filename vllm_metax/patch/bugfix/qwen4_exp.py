# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Qwen4-only graph defaults and loaded-QSA startup warmup for vLLM 0.29.

No model-layer extension point exists for these two startup hooks. Preserve
the upstream implementation and restrict added behavior to Qwen4 owners.
"""

import sys
from vllm_metax.patch.utils import PATCH_MARKER, patch
from vllm.transformers_utils.config import _CONFIG_REGISTRY
from vllm_metax.models.qwen4_exp.config import Qwen4ExpConfig, Qwen4ExpTextConfig

# AutoConfig.register alone cannot override vLLM's package-owned Qwen4 configs.
# Replace only the two Qwen4 registry values; preserve all other model configs.
_CONFIG_REGISTRY.update(qwen4_exp=Qwen4ExpConfig, qwen4_exp_text=Qwen4ExpTextConfig)

_GRAPH_TARGET = "vllm.config.vllm.default_breakable_cudagraph_architectures"
_WARMUP_TARGET = "vllm.model_executor.warmup.kernel_warmup.kernel_warmup"
_ARCHITECTURES = frozenset(
    {"Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration", "Qwen4ExpMTP"}
)


@patch("vllm.config.vllm", "default_breakable_cudagraph_architectures")
def default_breakable_cudagraph_architectures():
    original = getattr(default_breakable_cudagraph_architectures, PATCH_MARKER)[
        _GRAPH_TARGET
    ]
    return original() | _ARCHITECTURES


@patch("vllm.model_executor.warmup.kernel_warmup", "kernel_warmup")
def kernel_warmup(worker, *, process_local_only: bool = False):
    original = getattr(kernel_warmup, PATCH_MARKER)[_WARMUP_TARGET]
    original(worker, process_local_only=process_local_only)
    if "vllm_metax.models.qwen4_exp.metax.qsa" in sys.modules:
        from vllm_metax.models.qwen4_exp.metax.warmup import warmup_loaded_qsa

        warmup_loaded_qsa(worker)
