# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Keep model-dtype activations for dynamically quantized INT8 MLA weights. Patch
#     the shared dtype helper to cover both context paths while preserving upstream
#     chunking, parallelism and FP8 handling.
#
# Affected versions: vLLM 0.29.1.dev0 (98dff2a81d), verified 2026-09-17.
#
# Remove at: Upstream _get_kv_b_proj_input_dtype preserves model-dtype activations for
#     dynamically quantized INT8 weights.
# -----------------------------------------------------------------------------

"""Keep model-dtype activations for dynamically quantized INT8 MLA weights.

Affected: vLLM 0.29.1.dev0 (98dff2a81d). Remove when the upstream dtype helper
excludes INT8 weights. Patching the shared helper covers both context paths
without replacing MLA's chunking, parallelism, or FP8 handling.
"""

import torch
from vllm.model_executor.layers.attention.mla_attention import (
    _get_kv_b_proj_input_dtype as _original_get_input_dtype,
)
from vllm_metax.patch.utils import patch


@patch("vllm.model_executor.layers.attention.mla_attention")
def _get_kv_b_proj_input_dtype(kv_b_proj, use_fp8_prefill: bool) -> torch.dtype | None:
    # /-------------------- MetaX Modification --------------------\
    weight = getattr(kv_b_proj, "weight", None)
    if weight is not None and weight.dtype == torch.int8:
        return None
    # \-------------------- MetaX Modification --------------------/
    return _original_get_input_dtype(kv_b_proj, use_fp8_prefill)
