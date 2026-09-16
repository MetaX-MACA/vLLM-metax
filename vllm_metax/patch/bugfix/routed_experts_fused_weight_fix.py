# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd.
# All Rights Reserved.
# -----------------------------------------------
# Note: Fix fused weight orientation checks in RoutedExperts.load_weights.
#       Upstream incorrectly transposes per-channel scale tensors whose
#       dimensions do not contain the hidden size.
#
#       Upstream PR: https://github.com/vllm-project/vllm/pull/50137
#
# Affected versions: v0.25.0
#
# Remove at: Remove after the equivalent of vLLM PR #50137 is included in
#            the minimum supported vLLM version.
# -----------------------------------------------

from collections.abc import Iterable

import torch

from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts


def _orient_fused_weight(
    fused_weight: torch.Tensor,
    shard_id: str,
    unpadded_hidden: int,
) -> torch.Tensor:
    """Orient fused weights while leaving per-channel scales unchanged."""
    # Per-channel scale tensors (e.g. w13_weight_scale) have shape
    # [E, intermediate, 1] or [E, 1, intermediate], where the dimension
    # of size 1 corresponds to the per-channel reduced axis.  The
    # vLLM-native layout keeps the singleton on the last axis
    # ([E, intermediate, 1]); the transposed checkpoint layout puts it on
    # the middle axis ([E, 1, intermediate]) and must be normalized so the
    # later chunk(2, dim=1) splits the w1/w3 halves instead of chunking the
    # singleton dimension (which yields empty [E, 0, ...] slices and a
    # copy_ shape mismatch).
    if fused_weight.shape[-1] == 1 or fused_weight.shape[-2] == 1:
        if fused_weight.shape[-2] == 1 and fused_weight.shape[-1] != 1:
            return fused_weight.transpose(-1, -2)
        return fused_weight

    if shard_id == "w2":
        hidden_axis, intermediate_axis = -2, -1
    else:
        hidden_axis, intermediate_axis = -1, -2

    if (
        fused_weight.shape[hidden_axis] != unpadded_hidden
        and fused_weight.shape[intermediate_axis] == unpadded_hidden
    ):
        return fused_weight.transpose(-1, -2)
    return fused_weight


def _patched_load_weights(
    self,
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[str]:
    """Load fused experts without transposing quantization scales."""
    expert_mapping = self.get_expert_mapping(include_fused=True)
    unpadded_hidden = self.moe_config.hidden_dim_unpadded or self.moe_config.hidden_dim
    for expert_name, loaded_weight in weights:
        qual_name = f"{self.layer_name}.{expert_name}"
        is_fused = loaded_weight.dim() == 3
        matched = False
        for param_name, weight_name, expert_id, shard_id in expert_mapping:
            if weight_name not in qual_name:
                if matched and is_fused:
                    break
                continue
            matched = True
            weight_name = qual_name.replace(weight_name, param_name)
            param_name = weight_name.removeprefix(f"{self.layer_name}.")
            param = getattr(self, param_name)
            if is_fused:
                # /-------------------- Metax Modification ---------------------\
                fused_weight = _orient_fused_weight(
                    loaded_weight,
                    shard_id,
                    unpadded_hidden,
                )
                # \-------------------------------------------------------------/
                if shard_id in {"w1", "w3"}:
                    experts_shard = fused_weight.chunk(2, dim=1)[expert_id]
                else:
                    experts_shard = fused_weight
                start = 0
            else:
                experts_shard = loaded_weight.unsqueeze(0)
                start = expert_id

            loaded_experts = experts_shard.unbind()
            for local_expert_id, loaded_expert in enumerate(
                loaded_experts,
                start=start,
            ):
                success = param.weight_loader(
                    param=param,
                    loaded_weight=loaded_expert,
                    weight_name=weight_name,
                    shard_id=shard_id,
                    expert_id=local_expert_id,
                    return_success=True,
                )
                if success:
                    yield param_name


if not hasattr(RoutedExperts, "_orient_fused_weight"):
    # /-------------------- Metax Modification ---------------------\
    RoutedExperts._metax_original_load_weights = RoutedExperts.load_weights
    RoutedExperts._orient_fused_weight = staticmethod(_orient_fused_weight)
    RoutedExperts.load_weights = _patched_load_weights
    # \-------------------------------------------------------------/
