# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: logic error for loading experts.w13.scale
#
# Affected versions: v0.26.0
#
# Remove at: Once upstream is fixed.
# -----------------------------------------------------------------------------

from typing import Iterable
import torch

from vllm_metax.patch.utils import patch
from vllm.model_executor.layers.fused_moe.routed_experts import logger


@patch(
    "vllm.model_executor.layers.fused_moe.routed_experts", "RoutedExperts.load_weights"
)
def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> Iterable[str]:
    expert_mapping = self.get_expert_mapping(include_fused=True)
    unpadded_hidden = self.moe_config.hidden_dim_unpadded
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
                # w1 and w3 share one fused tensor; use a local copy so the
                # transpose below doesn't mutate loaded_weight across
                # iterations (else w3 is transposed twice and wrongly chunked)
                fused_weight = loaded_weight
                # /-------------------- MetaX Modification --------------------\
                # Fix logic error for loading experts.w13.scale
                if shard_id in {"w1", "w3"}:
                    if "scale" in weight_name:
                        # [experts, gate_up, 1]
                        experts_shard = loaded_weight.chunk(2, dim=1)[expert_id]
                    else:
                        if fused_weight.shape[-1] != unpadded_hidden:
                            # [..., hidden, intermediate] -> [..., intermediate, hidden]
                            fused_weight = fused_weight.transpose(-1, -2)
                    # Repurpose expert_id for deconcatenating w1 and w3
                    experts_shard = fused_weight.chunk(2, dim=1)[expert_id]
                # \------------------------------------------------------------/
                else:
                    if fused_weight.shape[-2] != unpadded_hidden:
                        # [..., intermediate, hidden] -> [..., hidden, intermediate]
                        fused_weight = fused_weight.transpose(-1, -2)
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
