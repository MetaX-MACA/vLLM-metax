# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch
from vllm.logger import logger
from typing import Callable, Optional, Union
from vllm.model_executor.layers.fused_moe import FusedMoE
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from torch.nn import Module
from vllm.model_executor.layers.fused_moe import FusedMoEConfig
from vllm.model_executor.utils import set_weight_attrs
from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
    find_matched_target,
    should_ignore_layer,
    _find_first_match,
    _match_fused_layer,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    check_moe_marlin_supports_layer,
)
from compressed_tensors.quantization import ActivationOrdering, QuantizationStrategy
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (
    CompressedTensorsMoEMethod,
    CompressedTensorsW8A8Int8MoEMethod,
    CompressedTensorsWNA16MoEMethod,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    GroupShape,
    FusedMoEQuantDesc,
)
from vllm_metax.ops.fused_moe import MacaUnquantizedFusedMoEMethod


def find_moe_matched_target(
    layer_name: Optional[str],
    module: Module,
    targets: Iterable[str],
    fused_mapping: Mapping[str, list[str]] = MappingProxyType({}),
) -> str | None:
    if layer_name is None:
        layer_name = ""

    matched_target = (
        _find_first_match(layer_name, targets)
        or _find_first_match(module.__class__.__name__, targets, True)
        or _match_fused_layer(layer_name, targets, fused_mapping)
    )

    return matched_target


class MacaCompressedTensorsMoEMethod(CompressedTensorsMoEMethod):
    @staticmethod
    def get_moe_method(
        quant_config: "CompressedTensorsConfig",  # type: ignore # noqa E501
        layer: torch.nn.Module,
        prefix: str,
    ) -> "CompressedTensorsMoEMethod":
        # are supported + check if the layer is being ignored.
        if should_ignore_layer(
            prefix, quant_config.ignore, quant_config.packed_modules_mapping
        ):
            return MacaUnquantizedFusedMoEMethod(layer.moe_config)

        # Check if one of the moe layers are defined in quant_config
        matched_moe_target = find_moe_matched_target(
            layer_name=prefix,
            module=layer,
            targets=quant_config.target_scheme_map.keys(),
            fused_mapping=quant_config.packed_modules_mapping,
        )

        if matched_moe_target is not None:
            matched_target = matched_moe_target
        elif "Linear" in quant_config.target_scheme_map:
            matched_target = "Linear"
        else:
            # May have instead defined the linear layers in the fused model
            fused_layers = ["re:.*down_proj.*", "re:.*gate_proj.*", "re:.*up_proj.*"]
            current_scheme = None
            for fused_layer in fused_layers:
                # Check if one of the fused layers are defined in quant_config
                matched_target = find_matched_target(
                    layer_name=fused_layer,
                    module=layer,
                    targets=quant_config.target_scheme_map.keys(),
                    fused_mapping=quant_config.packed_modules_mapping,
                )

                # Only valid if down_proj, gate_proj, and up_proj
                # are mapped to the same quant scheme in the quant_config
                if current_scheme is None:
                    current_scheme = quant_config.target_scheme_map.get(matched_target)
                else:
                    assert current_scheme == quant_config.target_scheme_map.get(
                        matched_target
                    )

        weight_quant = quant_config.target_scheme_map[matched_target].get("weights")
        input_quant = quant_config.target_scheme_map[matched_target].get(
            "input_activations"
        )

        if quant_config._is_wNa16_group_channel(weight_quant, input_quant):
            # group_size=None means channelwise
            group_size = weight_quant.group_size or -1
            # Prefer to use the MarlinMoE kernel when it is supported.
            if not check_moe_marlin_supports_layer(layer, group_size):
                if (
                    weight_quant.strategy in QuantizationStrategy.GROUP
                    and weight_quant.actorder
                    in (ActivationOrdering.GROUP, ActivationOrdering.DYNAMIC)
                ):
                    raise ValueError(
                        "WNA16MoE is not supported with actorder=group/dynamic."
                    )
                logger.info_once("Using CompressedTensorsWNA16MoEMethod")
                return MacaCompressedTensorsWNA16MoEMethod(
                    quant_config, layer.moe_config
                )
            else:
                raise RuntimeError(
                    "Unsupported FusedMoe scheme: CompressedTensorsWNA16MarlinMoEMethod"
                )
        elif quant_config._is_fp4a4_nvfp4(weight_quant, input_quant):
            return MacaCompressedTensorsWNA16MoEMethod(layer.moe_config)
        elif quant_config._is_dynamic_token_w8a8(weight_quant, input_quant):
            return MacaCompressedTensorsW8A8Int8MoEMethod(
                quant_config, layer.moe_config
            )
        elif quant_config._is_dynamic_token_w4a8_int(weight_quant, input_quant):
            logger.info_once("Using MacaCompressedTensorsW4A8Int4MoEMethod")
            return MacaCompressedTensorsW4A8Int4MoEMethod(
                quant_config, layer.moe_config, matched_target
            )
        else:
            raise RuntimeError(
                f"Unsupported FusedMoe scheme: {weight_quant}, {input_quant}"
            )


class MacaCompressedTensorsW8A8Int8MoEMethod(CompressedTensorsW8A8Int8MoEMethod):
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
        if enable_eplb:
            raise NotImplementedError(
                "EPLB not supported for `CompressedTensorsW8A8Int8MoEMethod` yet."
            )

        from vllm_metax.model_executor.layers.fused_moe import fused_experts

        topk_weights, topk_ids, _ = FusedMoE.select_experts(
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
            indices_type=self.topk_indices_dtype,
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
        if enable_eplb:
            raise NotImplementedError(
                "EPLB not supported for `CompressedTensorsWNA16MoEMethod` yet."
            )

        from vllm_metax.model_executor.layers.fused_moe import fused_experts

        topk_weights, topk_ids, _ = FusedMoE.select_experts(
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
            indices_type=self.topk_indices_dtype,
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


def int4_w4a8_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w1_zp: Optional[torch.Tensor],
    w2_zp: Optional[torch.Tensor],
    block_shape: Optional[list[int]] = None,
) -> FusedMoEQuantConfig:
    """
    Construct a quant config for 8-bit int activations and int4 weights.
    Note: Activations are pre-quantized.
    """
    group_shape = GroupShape(*block_shape) if block_shape is not None else None
    return FusedMoEQuantConfig(
        # torch.int8,
        _a1=FusedMoEQuantDesc("int8", shape=group_shape),
        _a2=FusedMoEQuantDesc("int8", shape=group_shape),
        _w1=FusedMoEQuantDesc("int4", group_shape, w1_scale, None, w1_zp),
        _w2=FusedMoEQuantDesc("int4", group_shape, w2_scale, None, w2_zp),
    )


class MacaCompressedTensorsW4A8Int4MoEMethod(CompressedTensorsMoEMethod):
    """
    CPU-only MoE method using dynamic 4-bit matmul kernels on Arm Platform
    - Weights: int4 (stored as int8 values in [-8,7], packed to uint8 nibbles)
    - Scales: Fp32 for Channelwise , bf16 for groupwise quantization
    - Bias: Same data type as original weights
    - Activations: FP32/Bf16 dynamic per-token (A8 Int),
      quantized inside the kernel
    """

    def __init__(
        self,
        quant_config: "CompressedTensorsConfig",  # type: ignore # noqa E501
        moe: FusedMoEConfig,
        matched_target,
    ):
        super().__init__(moe)
        self.has_bias = self.moe.has_bias
        self.quant_config = quant_config

        # Validate scheme: weights=W4 (channel or group),
        # activations=dynamic TOKEN (A8)
        self.target = "Linear"
        if matched_target != "":
            self.target = matched_target

        self.weight_quant = self.quant_config.target_scheme_map[self.target].get(
            "weights"
        )
        self.input_quant = self.quant_config.target_scheme_map[self.target].get(
            "input_activations"
        )

        # Must be dynamic per-token activations
        if (
            self.input_quant.strategy != QuantizationStrategy.TOKEN
            or not self.input_quant.dynamic
        ):
            raise ValueError(
                "W4A8-int MoE needs dynamic per-token activation quantization."
            )

        # Weight can be channel-wise (group_size=None) or group-wise
        self.group_size = (
            self.weight_quant.group_size
            if (self.weight_quant.group_size is not None)
            else -1
        )
        if self.weight_quant.num_bits != 4:
            raise ValueError("This method only supports 4-bit weights (num_bits=4).")

        self.static_input_scales = False  # always dynamic per token

    # ---- parameter creation ----
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        pack_factor = 32 // 4
        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )

        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // pack_factor,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )

        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size // self.group_size if self.group_size else 1,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)

        w2_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                hidden_size,
                intermediate_size_per_partition // self.group_size
                if self.group_size
                else 1,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        extra_weight_attrs.update({"quant_method": self.weight_quant.strategy})
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.static_input_scales:
            raise ValueError(
                "For w4a8 Fused MoE layers, only dynamic scales"
                "for activations are supported. Found "
                f"{self.input_quant}"
            )
        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Reconfigure scales to match mctlass required format
        layer.w13_weight_scale = torch.nn.Parameter(
            layer.w13_weight_scale.transpose(1, 2).contiguous(), requires_grad=False
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            layer.w2_weight_scale.transpose(1, 2).contiguous(), requires_grad=False
        )

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> Optional[FusedMoEQuantConfig]:
        return int4_w4a8_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w1_zp=None,
            w2_zp=None,
            block_shape=[0, self.group_size],
        )

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
    ) -> torch.Tensor:
        assert not enable_eplb, "EPLB not supported for W4A8-int MoE yet."
        assert activation in ("silu", "swigluoai", "swiglu"), (
            "Only SiLU/SwiGLUGU/SwiGLUUG are supported."
        )
        assert expert_map is None, """expert_map/EP not implemented
        for CPU dyn-4bit MoE."""

        assert self.fused_experts is None

        if enable_eplb:
            raise NotImplementedError(
                "EPLB not supported for `CompressedTensorsW8A8Int8MoEMethod` yet."
            )

        from vllm_metax.model_executor.layers.fused_moe import fused_experts

        topk_weights, topk_ids, _ = FusedMoE.select_experts(
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
            indices_type=self.topk_indices_dtype,
        )

        return fused_experts(
            hidden_states=x,
            w1=layer.w13_weight_packed,
            w2=layer.w2_weight_packed,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            use_int4_w4a8=True,
            quant_config=self.moe_quant_config,
        )
