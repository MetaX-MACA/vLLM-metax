# SPDX-License-Identifier: Apache-2.0
import torch
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors_moe as vllm_ctm,
)

# -----------------------------------------------------------
# Note: We need to keep the method name **the same** as vLLM's
# -----------------------------------------------------------


class CompressedTensorsMoEMethod(vllm_ctm.CompressedTensorsMoEMethod):
    @staticmethod
    def get_moe_method(
        quant_config: "CompressedTensorsConfig",  # type: ignore # noqa E501
        layer: torch.nn.Module,
        layer_name: str,
    ) -> "CompressedTensorsMoEMethod":
        origin_moe_method = vllm_ctm.CompressedTensorsMoEMethod.get_moe_method(
            quant_config, layer, layer_name
        )

        # -------------------------------------------
        # Note: all these are copied from vllm's logic
        #       we just need the
        #           - `weight_quant`
        #           - `input_quant`
        #       to construct the corresponding class.
        quant_config._add_fused_moe_to_target_scheme_map()
        unfused_names = [
            layer_name + proj_name
            for proj_name in [".0.gate_proj", ".0.up_proj", ".0.down_proj"]
        ]
        all_scheme_dicts = [
            quant_config.get_scheme_dict(layer, name) for name in unfused_names
        ]
        scheme_dict = all_scheme_dicts.pop()
        if not all([cur_dict == scheme_dict for cur_dict in all_scheme_dicts]):
            raise ValueError(
                "All MoE projections need to have same "
                "quantization scheme but found multiple"
            )
        weight_quant = scheme_dict.get("weights")
        input_quant = scheme_dict.get("input_activations")

        # -------------------------------------------
        # Replace with Metax's MoE quantization methods by:
        #  - `weights_quant`
        #  - `input_quant`
        if isinstance(origin_moe_method, vllm_ctm.CompressedTensorsWNA16MoEMethod):
            return CompressedTensorsWNA16MoEMethod(
                weight_quant, input_quant, layer.moe_config
            )
        elif isinstance(origin_moe_method, vllm_ctm.CompressedTensorsW8A8Int8MoEMethod):
            return CompressedTensorsW8A8Int8MoEMethod(
                weight_quant, input_quant, layer.moe_config
            )


class CompressedTensorsW8A8Int8MoEMethod(vllm_ctm.CompressedTensorsW8A8Int8MoEMethod):
    def apply(
        self, layer: FusedMoE, x: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # here we use plugin's `fused_experts`
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
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            quant_config=self.moe_quant_config,
        )


class CompressedTensorsWNA16MoEMethod(vllm_ctm.CompressedTensorsWNA16MoEMethod):
    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # here we use plugin's `fused_experts`
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
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            quant_config=self.moe_quant_config,
        )
