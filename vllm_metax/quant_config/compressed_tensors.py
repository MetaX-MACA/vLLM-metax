# SPDX-License-Identifier: Apache-2.0

from typing import Optional, Union, Callable

import torch
from compressed_tensors.quantization import (QuantizationStrategy)

from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization.base_config import (  # noqa: E501
    QuantizeMethodBase)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    CompressedTensorsMoEMethod)
from vllm.model_executor.layers.quantization.compressed_tensors.transform.linear import (  # noqa: E501
    CompressedTensorsLinearTransformMethod, get_linear_transform_schemes)
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    find_matched_target)

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig, CompressedTensorsLinearMethod,
    CompressedTensorsKVCacheMethod)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
    logger, CompressedTensorsWNA16MoEMethod, CompressedTensorsW4A4MoeMethod,
    CompressedTensorsW8A8Fp8MoEMethod, CompressedTensorsW8A8Int8MoEMethod)
from vllm_metax.patch.model_executor.hook_register import (
    register_quantization_config)

from compressed_tensors.quantization import (ActivationOrdering)

__all__ = [
    "MacaCompressedTensorsW8A8Int8MoEMethod",
]


@register_quantization_config("compressed-tensors")
class MacaCompressedTensorsConfig(CompressedTensorsConfig):

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import

        if isinstance(layer, LinearBase):
            # collect schemes
            quant_scheme = self.get_scheme(layer=layer, layer_name=prefix)
            input_tfms, output_tfms = get_linear_transform_schemes(
                layer, prefix, self.transform_config,
                self.packed_modules_mapping)

            # choose quantization method
            quant_method: LinearMethodBase = UnquantizedLinearMethod()
            if quant_scheme is not None:
                layer.scheme = quant_scheme
                quant_method = CompressedTensorsLinearMethod(self)

            # choose transform method
            if any((input_tfms, output_tfms)):
                return CompressedTensorsLinearTransformMethod.from_schemes(
                    quant_method, input_tfms, output_tfms)

            else:
                return quant_method

        if isinstance(layer, Attention):
            return CompressedTensorsKVCacheMethod(self)
        if isinstance(layer, FusedMoE):
            return MacaCompressedTensorsMoEMethod.get_moe_method(self, layer)
        return None


class MacaCompressedTensorsMoEMethod(CompressedTensorsMoEMethod):

    @staticmethod
    def get_moe_method(
        quant_config: "CompressedTensorsConfig",  # type: ignore # noqa E501
        layer: torch.nn.Module
    ) -> "CompressedTensorsMoEMethod":
        # TODO: @dsikka: refactor this to use schemes as other kernels
        # are supported + check if the layer is being ignored.
        # Check if a using "Linear" to select schemes
        if "Linear" in quant_config.target_scheme_map:
            matched_target = "Linear"
        else:
            # May have instead defined the linear layers in the fused model

            fused_layers = [
                "re:.*down_proj.*", "re:.*gate_proj.*", "re:.*up_proj.*"
            ]
            current_scheme = None
            for fused_layer in fused_layers:
                # Check if one of the fused layers are defined in quant_config
                matched_target = find_matched_target(
                    layer_name=fused_layer,
                    module=layer,
                    targets=quant_config.target_scheme_map.keys(),
                    fused_mapping=quant_config.packed_modules_mapping)

                # Only valid if down_proj, gate_proj, and up_proj
                # are mapped to the same quant scheme in the quant_config
                if current_scheme is None:
                    current_scheme = quant_config.target_scheme_map.get(
                        matched_target)
                else:
                    assert current_scheme == quant_config.target_scheme_map.get(
                        matched_target)

        weight_quant = quant_config.target_scheme_map[matched_target].get(
            "weights")
        input_quant = quant_config.target_scheme_map[matched_target].get(
            "input_activations")

        if quant_config._is_wNa16_group_channel(weight_quant, input_quant):
            # Not to use the MarlinMoE kernel.
            if (weight_quant.strategy in QuantizationStrategy.GROUP
                    and weight_quant.actorder
                    in (ActivationOrdering.GROUP, ActivationOrdering.DYNAMIC)):
                raise ValueError(
                    "WNA16MoE is not supported with actorder=group/dynamic.")
            logger.info_once("Using CompressedTensorsWNA16MoEMethod")
            return MacaCompressedTensorsWNA16MoEMethod(quant_config,
                                                       layer.moe_config)
        elif quant_config._is_fp4a4_nvfp4(weight_quant, input_quant):
            return CompressedTensorsW4A4MoeMethod(layer.moe_config, layer)
        elif (quant_config._is_fp8_w8a8_sm90(weight_quant, input_quant)
              or quant_config._is_fp8_w8a8_sm100(weight_quant, input_quant)
              or quant_config._is_fp8_w8a8(weight_quant, input_quant)):
            return CompressedTensorsW8A8Fp8MoEMethod(quant_config,
                                                     layer.moe_config)
        elif quant_config._is_dynamic_token_w8a8(weight_quant, input_quant):
            return MacaCompressedTensorsW8A8Int8MoEMethod(
                quant_config, layer.moe_config)
        else:
            raise RuntimeError(
                f"Unsupported FusedMoe scheme: {weight_quant}, {input_quant}")


class MacaCompressedTensorsW8A8Int8MoEMethod(CompressedTensorsW8A8Int8MoEMethod
                                             ):

    def apply(
        self,
        layer: torch.nn.Module,
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
        assert self.fused_experts is None

        if enable_eplb:
            raise NotImplementedError(
                "EPLB not supported for "
                "`CompressedTensorsW8A8Int8MoEMethod` yet.")

        from vllm_metax.model_executor.layers.fused_moe import fused_experts

        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            indices_type=self.topk_indices_dtype)

        return fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            use_int8_w8a8=True,
            per_channel_quant=True,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale)


class MacaCompressedTensorsWNA16MoEMethod(CompressedTensorsWNA16MoEMethod):

    def apply(
        self,
        layer: torch.nn.Module,
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
        assert self.fused_experts is None

        if enable_eplb:
            raise NotImplementedError("EPLB not supported for "
                                      "`CompressedTensorsWNA16MoEMethod` yet.")

        from vllm_metax.model_executor.layers.fused_moe import fused_experts

        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
            e_score_correction_bias=e_score_correction_bias,
            indices_type=self.topk_indices_dtype)

        return fused_experts(
            x,
            layer.w13_weight_packed,
            layer.w2_weight_packed,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=activation,
            use_int4_w4a16=self.num_bits == 4,
            use_int8_w8a16=self.num_bits == 8,
            global_num_experts=global_num_experts,
            apply_router_weight_on_input=apply_router_weight_on_input,
            expert_map=expert_map,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w1_zp=None,
            w2_zp=None,
            block_shape=[0, self.group_size])
