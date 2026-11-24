# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

"""MACA OOT override for QwenGatedDeltaNetAttention.

Adds a guard to the fused MTP decode fast-path selection: the
``fused_gdn_decode_post_conv_mtp`` op assumes a SiLU output gate, so for
sigmoid-gated checkpoints (e.g. Qwen3.8-Flash-Next, ``output_gate_type:
"sigmoid"``) the fused path must be rejected and the non-fused MTP decode
used instead. Without the guard, serving a sigmoid-gated model with MTP
speculative decoding either crashes (public mcoplib 0.4.13 exposes a
14-argument op while the vllm 0.30 wrapper passes 15) or silently corrupts
the decode output.
"""

import torch

from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)


@QwenGatedDeltaNetAttention.register_oot
class MacaQwenGatedDeltaNetAttention(QwenGatedDeltaNetAttention):
    def _can_use_fused_gdn_mtp_decode(self, attn_metadata) -> bool:
        can_use = super()._can_use_fused_gdn_mtp_decode(attn_metadata)
        if not can_use:
            return False
        # The fused MTP decode op assumes a SiLU output gate; sigmoid-gated
        # checkpoints must take the non-fused MTP path (see class docstring).
        return self.norm.activation == "silu"
