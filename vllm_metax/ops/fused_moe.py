# SPDX-License-Identifier: Apache-2.0
from vllm.model_executor.layers.fused_moe.layer import (
    UnquantizedFusedMoEMethod,
)

from typing import Callable
import torch

from vllm_metax.model_executor.layers.fused_moe.fused_moe import fused_experts


@UnquantizedFusedMoEMethod.register_oot
class MacaUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    def forward_oot(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: torch.Tensor | None = None,
        logical_to_physical_map: torch.Tensor | None = None,
        logical_replica_count: torch.Tensor | None = None,
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
            activation=activation,
            quant_config=self.moe_quant_config,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
        )
        if layer.zero_expert_num != 0 and layer.zero_expert_type is not None:
            assert not isinstance(result, tuple), (
                "Shared + zero experts are mutually exclusive not yet supported"
            )
            return result, zero_expert_result
        else:
            return result
