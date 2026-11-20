# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Keep metadata-dependent graph breaks local to the Qwen4 model."""

import torch
from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)


class Qwen4ExpGatedDeltaNetAttention(QwenGatedDeltaNetAttention):
    @eager_break_during_capture
    def _forward_core(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
    ):
        return super()._forward_core(mixed_qkv, b, a, core_attn_out)

    @eager_break_during_capture
    def _forward_core_fused_norm_packed(
        self,
        mixed_qkvz: torch.Tensor,
        ba: torch.Tensor,
        core_attn_out: torch.Tensor,
    ) -> None:
        return super()._forward_core_fused_norm_packed(mixed_qkvz, ba, core_attn_out)
