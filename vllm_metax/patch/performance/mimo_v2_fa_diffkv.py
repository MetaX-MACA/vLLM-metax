# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Ensure vllm-metax attention backends are registered before the upstream
#     MiMo-V2 attention layer selects its DiffKV backend.  Without this, the
#     enum still resolves FLASH_ATTN_DIFFKV to the upstream vLLM class, whose
#     sink check is false on MACA FlashAttention v2; the model silently falls
#     back to the much slower TRITON_ATTN_DIFFKV path.
#
# Affected versions: vLLM 0.29.x with the MiMo-V2 Flash model
#     (vllm.model_executor.models.mimo_v2.MiMoV2Attention).  Verified on
#     2026-10-26 with MiMo-V2-Flash-FP8, tp4/pp2, MACA 3.9.0.31.
#
# Remove at: upstream vLLM selects the registered vendor backend before model
#     construction, or provides an explicit attention-backend extension point
#     for DiffKV sinks.
# -----------------------------------------------------------------------------

from __future__ import annotations

from vllm.model_executor.models.mimo_v2 import MiMoV2Attention
from vllm_metax.patch.utils import patch
from vllm_metax.platform import register_attention_backends

# Capture the original before @patch replaces the class attribute.  The wrapper
# only adds the early registration; the upstream selection logic is untouched.
_ORIGINAL_INIT = MiMoV2Attention.__init__
_BACKENDS_REGISTERED = False


@patch("vllm.model_executor.models.mimo_v2", "MiMoV2Attention.__init__")
def mimo_v2_attention_init(self, *args, **kwargs):
    global _BACKENDS_REGISTERED
    # /-------------------- MetaX Modification --------------------\
    # The MiMo layer builds Attention in __init__; register the vendor
    # FlashAttentionDiffKV backend first so the enum lookup below sees the
    # MetaX implementation instead of the upstream one.
    if not _BACKENDS_REGISTERED:
        register_attention_backends()
        _BACKENDS_REGISTERED = True
    # \-------------------- MetaX Modification --------------------/
    return _ORIGINAL_INIT(self, *args, **kwargs)
