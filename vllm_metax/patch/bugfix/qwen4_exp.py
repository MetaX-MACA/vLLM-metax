# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Select MetaX Qwen4 config subclasses to normalize the checkpoint layer
#     alias "qwen_sparse_attention" to "full_attention" before upstream config
#     validation. Append startup compilation for loaded MetaX QSA layers because
#     upstream QSA warmup discovers only vllm.models.qwen4_exp.nvidia modules.
#
# Affected versions: Source-reviewed against vLLM 0.30.0 (ced6857afa0e),
#     2026-10-16; applies to Qwen4 checkpoints using the alias and MetaX QSA models.
#
# Remove at: Remove the config override when upstream normalizes the alias for
#     this model path, or migrate it when a public config-class override API is
#     available. Remove the warmup wrapper when upstream startup warmup reaches
#     the MetaX QSA kernels through a supported registration hook.
# -----------------------------------------------------------------------------
"""Qwen4 checkpoint config compatibility and MetaX QSA startup compilation."""

import sys
from vllm_metax.patch.utils import PATCH_MARKER, patch
from vllm.transformers_utils.config import _CONFIG_REGISTRY
from vllm_metax.models.qwen4_exp.config import Qwen4ExpConfig, Qwen4ExpTextConfig

# /-------------------- MetaX Modification --------------------\
# HFConfigParser registers classes from this table with exist_ok=True during
# loading. AutoConfig.register / _register_config_class alone would be overwritten
# by that registration, so replace both the outer and standalone text configs.
# The outer MetaX config also selects the MetaX text class for nested text_config.
_CONFIG_REGISTRY.update(qwen4_exp=Qwen4ExpConfig, qwen4_exp_text=Qwen4ExpTextConfig)
# \-------------------- MetaX Modification --------------------/

_WARMUP_TARGET = "vllm.model_executor.warmup.kernel_warmup.kernel_warmup"


@patch("vllm.model_executor.warmup.kernel_warmup", "kernel_warmup")
def kernel_warmup(worker, *, process_local_only: bool = False):
    # Preserve upstream warmup and forward the process-local mode unchanged.
    original = getattr(kernel_warmup, PATCH_MARKER)[_WARMUP_TARGET]
    original(worker, process_local_only=process_local_only)
    # /-------------------- MetaX Modification --------------------\
    # Avoid importing QSA for unrelated models. The helper checks actual model
    # owners and deduplicates compilation by cache/table layout and head counts.
    # It compiles sparse attention variants; the dynamic topk selector still JITs
    # on demand. This addition also runs in process-local mode and currently does
    # not consult enable_jit_warmup, unlike upstream QSA warmup.
    if "vllm_metax.models.qwen4_exp.metax.qsa" in sys.modules:
        from vllm_metax.models.qwen4_exp.metax.warmup import warmup_loaded_qsa

        warmup_loaded_qsa(worker)
    # \-------------------- MetaX Modification --------------------/
