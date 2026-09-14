# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd.
# All Rights Reserved.
# -----------------------------------------------
# Note: Normalize the per-channel weight scales of transposed fused MoE
#       checkpoints before they are loaded (patches
#       ``RoutedExperts._orient_fused_weight``).
#
#       Qwen3-VL-MoE int8 W8A8 checkpoints (Qwen3-VL-235B-A22B-Instruct_W8A8,
#       Qwen3-VL-30B-A3B-Instruct_W8A8, ...) store the fused experts transposed
#       and store the per-channel weight scales in that same layout, so the
#       quantized (singleton) axis ends up in the middle:
#
#           mlp.experts.gate_up_proj        [E, hidden, 2*intermediate]
#           mlp.experts.gate_up_proj_scale  [E, 1, 2*intermediate]
#           mlp.experts.down_proj           [E, intermediate, hidden]
#           mlp.experts.down_proj_scale     [E, 1, hidden]
#
#       RoutedExperts.load_weights only transposes the fused weights, because
#       `uses_weight_layout` excludes scale tensors unless they are BLOCK
#       scales. The per-channel scales therefore reach the weight loaders with
#       the channel axis in the middle and loading fails:
#
#         - w2: RuntimeError: output with shape [1, 1] doesn't match the
#               broadcast shape [1, 4096] (routed_experts.py
#               _load_per_channel_weight_scale)
#         - w1/w3: chunk(2, dim=1) splits the singleton axis into [E, 1, N]
#               and [E, 0, N] and _load_w13 fails in the same way.
#
#       vLLM allocates per-channel MoE scales as [E, N, 1], so moving the
#       singleton axis back to the last position fixes both paths. The check is
#       shape based, which leaves native [E, N, 1] scales untouched.
#
# Purpose: port the v0.25.0/v0.26.0 fix of routed_experts_fused_weight_fix.py to
#          v0.28.0, where upstream only handles the fused weights.
#
# Affected versions: v0.28.0 (regression from v0.26.0)
#
# Remove at: Remove once upstream normalizes the per-channel scales of
#            transposed fused checkpoints (follow-up of vLLM PR #50137), or
#            once the affected checkpoints are re-quantized with native
#            [E, N, 1] scale layouts.
# -----------------------------------------------

import torch

from vllm_metax.patch.utils import patch


@patch(  # type: ignore[misc]
    "vllm.model_executor.layers.fused_moe",
    "RoutedExperts._orient_fused_weight",
)
@staticmethod
def _orient_fused_weight(
    fused_weight: torch.Tensor,
    is_fused_checkpoint_transposed: bool,
) -> torch.Tensor:
    """Normalise a fused expert tensor to the vLLM weight layout."""
    # /-------------------- Metax Modification ---------------------\
    # Transposed fused checkpoints also store per-channel scales transposed
    # ([E, 1, N]); upstream skips the transpose for them. See the note above.
    if fused_weight.shape[-2] == 1 and fused_weight.shape[-1] != 1:
        return fused_weight.transpose(-1, -2)
    # \-------------------------------------------------------------/
    if is_fused_checkpoint_transposed:
        return fused_weight.transpose(-1, -2)
    return fused_weight
