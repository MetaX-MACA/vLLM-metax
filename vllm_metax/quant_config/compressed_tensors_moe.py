# SPDX-License-Identifier: Apache-2.0
import torch
from typing import Callable
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors_moe as vllm_ctm,
)

# -----------------------------------------------------------
# Note: We need to keep the name **consistent** with vLLM's
# -----------------------------------------------------------


class CompressedTensorsMoEMethod(vllm_ctm.CompressedTensorsMoEMethod):
    @staticmethod
    def get_moe_method(
        quant_config: "CompressedTensorsConfig",  # type: ignore # noqa E501
        layer: torch.nn.Module,
        layer_name: str,
    ) -> "CompressedTensorsMoEMethod":
        moe_method = vllm_ctm.CompressedTensorsMoEMethod.get_moe_method(
            quant_config, layer, layer_name
        )
        # Replace with Metax's MoE quantization methods
        if isinstance(moe_method, vllm_ctm.CompressedTensorsWNA16MoEMethod):
            moe_method = CompressedTensorsWNA16MoEMethod(
                quant_config, layer.moe_config, layer_name
            )
        elif isinstance(moe_method, vllm_ctm.CompressedTensorsW8A8Int8MoEMethod):
            moe_method = CompressedTensorsW8A8Int8MoEMethod(
                quant_config, layer.moe_config, layer_name
            )
        return moe_method


class CompressedTensorsW8A8Int8MoEMethod(vllm_ctm.CompressedTensorsW8A8Int8MoEMethod):
    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
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
        from vllm_metax.model_executor.layers.fused_moe import fused_experts

        topk_weights, topk_ids, _ = layer.select_experts(
            hidden_states=x,
            router_logits=router_logits,
        )

        return fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            quant_config=self.moe_quant_config,
        )


class CompressedTensorsWNA16MoEMethod(vllm_ctm.CompressedTensorsWNA16MoEMethod):
    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
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
        from vllm_metax.model_executor.layers.fused_moe import fused_experts

        topk_weights, topk_ids, _ = layer.select_experts(
            hidden_states=x,
            router_logits=router_logits,
        )

        return fused_experts(
            x,
            layer.w13_weight_packed,
            layer.w2_weight_packed,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            quant_config=self.moe_quant_config,
        )
