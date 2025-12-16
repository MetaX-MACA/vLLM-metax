# SPDX-License-Identifier: Apache-2.0
import torch
from typing import Callable, Optional, Union
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
    CompressedTensorsMoEMethod,
    CompressedTensorsW8A8Int8MoEMethod,
    CompressedTensorsWNA16MoEMethod,
)


class MacaCompressedTensorsMoEMethod(CompressedTensorsMoEMethod):
    @staticmethod
    def get_moe_method(
        quant_config: "CompressedTensorsConfig",  # type: ignore # noqa E501
        layer: torch.nn.Module,
        layer_name: str,
    ) -> "CompressedTensorsMoEMethod":
        moe_method = CompressedTensorsMoEMethod.get_moe_method(quant_config, layer, layer_name)
        if isinstance(moe_method, CompressedTensorsWNA16MoEMethod):
            moe_method = MacaCompressedTensorsWNA16MoEMethod(
                quant_config, layer.moe_config
            )
        elif isinstance(moe_method, CompressedTensorsW8A8Int8MoEMethod):
            moe_method = MacaCompressedTensorsW8A8Int8MoEMethod(
                quant_config, layer.moe_config
            )
        return moe_method


class MacaCompressedTensorsW8A8Int8MoEMethod(CompressedTensorsW8A8Int8MoEMethod):
    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if enable_eplb:
            raise NotImplementedError(
                "EPLB not supported for `CompressedTensorsW8A8Int8MoEMethod` yet."
            )

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


class MacaCompressedTensorsWNA16MoEMethod(CompressedTensorsWNA16MoEMethod):
    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
        enable_eplb: bool = False,
        expert_load_view: Optional[torch.Tensor] = None,
        logical_to_physical_map: Optional[torch.Tensor] = None,
        logical_replica_count: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if enable_eplb:
            raise NotImplementedError(
                "EPLB not supported for `CompressedTensorsWNA16MoEMethod` yet."
            )

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
