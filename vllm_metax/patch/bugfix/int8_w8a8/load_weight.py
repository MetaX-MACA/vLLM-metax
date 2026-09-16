# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Fix fused weight orientation checks in RoutedExperts.load_weights.
#       v0.26.0 wrongly transposes fused per-channel scale tensors whose
#       dimensions do not contain the hidden size, and the transposed
#       checkpoint layout ([E, 1, 2*I]) must also be normalized before
#       chunk(2, dim=1). Otherwise w1/w3 splitting operates on the size-1
#       reduced axis and loading crashes in `_load_w13` with a copy_ shape
#       mismatch (e.g. [0, 1] vs [0, 3072]).
#
#       Mirrors upstream PR #50137 plus the transposed-scale normalization
#       from mx/v0.25.0-dev (987498d7).
#
# Affected versions: v0.26.0
#
# Remove at: Once upstream is fixed.
# -----------------------------------------------------------------------------

from typing import Iterable
import torch

from vllm_metax.patch.utils import patch
from vllm.model_executor.layers.fused_moe.routed_experts import logger


def _orient_fused_weight(
    fused_weight: torch.Tensor,
    shard_id: str,
    unpadded_hidden: int,
) -> torch.Tensor:
    """Orient fused weights while leaving per-channel scales unchanged."""
    # Per-channel scale tensors (e.g. experts.w13.scale) are
    # [E, 2*I, 1] in the vLLM-native layout and [E, 1, 2*I] in transposed
    # checkpoints. The dimension of size 1 is the per-channel reduced axis
    # and must stay last so the later chunk(2, dim=1) splits w1/w3 instead
    # of chunking the singleton (which yields an undivided [E, 1, 2*I]
    # slice that later TP-shards into [0, 2*I] and fails to copy).
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


@patch(
    "vllm.model_executor.layers.fused_moe.routed_experts", "RoutedExperts.load_weights"
)
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Iterable[str]:
    expert_mapping = self.get_expert_mapping(include_fused=True)
    unpadded_hidden = self.moe_config.hidden_dim_unpadded or self.moe_config.hidden_dim
    for expert_name, loaded_weight in weights:
        qual_name = f"{self.layer_name}.{expert_name}"
        # Fused expert weights can be identified by their 3D tensors
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
                # /-------------------- MetaX Modification --------------------\
                # Normalize fused weights and per-channel scales to the
                # canonical orientation before chunking w1/w3. Note that this
                # only returns views, so loaded_weight is never mutated across
                # iterations (else w3 would be transposed twice and wrongly
                # chunked).
                fused_weight = _orient_fused_weight(
                    loaded_weight,
                    shard_id,
                    unpadded_hidden,
                )
                # \------------------------------------------------------------/
                if shard_id in {"w1", "w3"}:
                    # Repurpose expert_id for deconcatenating w1 and w3
                    experts_shard = fused_weight.chunk(2, dim=1)[expert_id]
                else:
                    experts_shard = fused_weight
                start = 0
            else:
                # loaded_weight is a single expert weight, so we add a dummy expert
                # dimension to unify the loading logic with the fused case
                experts_shard = loaded_weight.unsqueeze(0)
                start = expert_id

            # Unified loading logic for fused and non-fused experts
            loaded_experts = experts_shard.unbind()
            for expert_id, loaded_expert in enumerate(loaded_experts, start=start):
                success = param.weight_loader(
                    param=param,
                    loaded_weight=loaded_expert,
                    weight_name=weight_name,
                    shard_id=shard_id,
                    expert_id=expert_id,
                    return_success=True,
                )
                if success:
                    logger.debug(
                        "Loaded expert %d of shard %s into %s for layer %s",
                        expert_id,
                        shard_id,
                        param_name,
                        self.layer_name,
                    )
                    yield param_name
