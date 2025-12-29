# SPDX-License-Identifier: Apache-2.0
from vllm.model_executor.layers.fused_moe.layer import (
    UnquantizedFusedMoEMethod,
)

import torch

from vllm_metax.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.platforms import current_platform


@UnquantizedFusedMoEMethod.register_oot
class MacaUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    def forward_oot(
        self,
        layer: "FusedMoe",  # type: ignore[name-defined] # noqa: F821
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        topk_weights, topk_ids, zero_expert_result = layer.select_experts(
            hidden_states=x,
            router_logits=router_logits,
        )

        result = fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=layer.activation,
            quant_config=self.moe_quant_config,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
        )

        if layer.zero_expert_num != 0 and layer.zero_expert_type is not None:
            assert not isinstance(result, tuple), (
                "Shared + zero experts are mutually exclusive not yet supported"
            )
            return result, zero_expert_result
        else:
            return result

    if current_platform.is_out_of_tree():
        forward_native = forward_oot
