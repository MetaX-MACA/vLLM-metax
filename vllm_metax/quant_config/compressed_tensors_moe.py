# SPDX-License-Identifier: Apache-2.0
import torch
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors_moe as vllm_ctm,
)
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    FusedMoEQuantDesc,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape

from vllm.model_executor.utils import set_weight_attrs


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
        # -------------------------------------------
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
        # -------------------------------------------
        if isinstance(
            origin_moe_method,
            (
                vllm_ctm.CompressedTensorsWNA16MoEMethod,
                vllm_ctm.CompressedTensorsWNA16MarlinMoEMethod,
            ),
        ):
            # -----------------------------------------------------------
            # We do not support CompressedTensors-MarlinMoEMethod currently
            # Fallback to non-Marlin methods
            # -----------------------------------------------------------
            return CompressedTensorsWNA16MoEMethod(
                weight_quant, input_quant, layer.moe_config
            )
        elif isinstance(origin_moe_method, vllm_ctm.CompressedTensorsW8A8Int8MoEMethod):
            return CompressedTensorsW8A8Int8MoEMethod(
                weight_quant, input_quant, layer.moe_config
            )
        elif isinstance(origin_moe_method, vllm_ctm.CompressedTensorsW4A8Int8MoEMethod):
            # --------------------------------------------------------------------
            # Note!: On maca W4A8 is hardware supported. The quantization scheme
            #       is selected by `quant_config._is_dynamic_token_w4a8_int`. So we
            #       just need to re-implement and map with Int4MoEMethod here.
            # --------------------------------------------------------------------
            return CompressedTensorsW4A8Int4MoEMethod(
                weight_quant, input_quant, layer.moe_config
            )
        return origin_moe_method


# -----------------------------------------------------------
# Note: We need to keep the method name **the same** as vLLM's
# -----------------------------------------------------------
class CompressedTensorsW8A8Int8MoEMethod(vllm_ctm.CompressedTensorsW8A8Int8MoEMethod):
    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # here we use Metax's `fused_experts`
        from vllm_metax.model_executor.layers.fused_moe.fused_moe import fused_experts

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


# -----------------------------------------------------------
# Note: We need to keep the method name **the same** as vLLM's
# -----------------------------------------------------------
class CompressedTensorsWNA16MoEMethod(vllm_ctm.CompressedTensorsWNA16MoEMethod):
    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ) -> mk.FusedMoEPermuteExpertsUnpermute:
        if self.moe.is_lora_enabled:
            assert self.moe_quant_config is not None
            from vllm.triton_utils import HAS_TRITON

            if HAS_TRITON:
                # use Metax's TritonWNA16Experts
                from vllm_metax.model_executor.layers.fused_moe.fused_moe import (
                    TritonWNA16Experts,
                )

                layer.w13_weight = layer.w13_weight_packed
                layer.w2_weight = layer.w2_weight_packed
                return TritonWNA16Experts(quant_config=self.moe_quant_config)
            else:
                raise NotImplementedError(
                    "TritonExperts requires Triton. "
                    "Install triton or disable LoRA for MoE."
                )

        raise NotImplementedError

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # here we use Metax's `fused_experts`
        from vllm_metax.model_executor.layers.fused_moe.fused_moe import fused_experts

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


class CompressedTensorsW4A8Int4MoEMethod(vllm_ctm.CompressedTensorsMoEMethod):
    """
    On maca W4A8 is hardware supported. But on maca the weights is
    packed to int32 nibbles instead of int8 values. Still we name it
    as `W4A8Int4`

    - Weights: int4 (packed to int32 nibbles)
    - Scales: Fp32 for Channelwise , bf16 for groupwise quantization
    - Bias: Same data type as original weights
    - Activations: FP32/Bf16 dynamic per-token (A8 Int),
      quantized inside the kernel
    """

    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
    ):
        super().__init__(moe)
        self.has_bias = self.moe.has_bias
        self.weight_quant = weight_quant
        self.input_quant = input_quant

        self.num_bits = self.weight_quant.num_bits
        self.packed_factor = 32 // self.num_bits

        # Validate scheme: weights=W4 (channel or group),
        # activations=dynamic TOKEN (A8)

        # Must be dynamic per-token activations
        if (
            input_quant.strategy != QuantizationStrategy.TOKEN
            or not input_quant.dynamic
        ):
            raise ValueError(
                "W4A8-int MoE needs dynamic per-token activation quantization."
            )

        # Weight can be channel-wise (group_size=None) or group-wise
        self.group_size = (
            weight_quant.group_size if (weight_quant.group_size is not None) else -1
        )
        if weight_quant.num_bits != 4:
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
        # pack_factor: 8 (int32 // 4 bits)
        # Shapes per local rank (TP/EP):
        #   w13: [E, 2*I_local, H // pack_factor]  int32  (pack together into int32 nibbles)
        #   w2 : [E, H, I_local // pack_factor]    int32  (pack together into int32 nibbles)
        # Scales:
        #   channel-wise: group_size=-1 -> per-output-row, single scale per row
        #   group-wise  : group_size=g   ->
        #   per-output-row, (in_features/g) scales

        E = num_experts
        H = hidden_size
        IN = intermediate_size_per_partition
        g = self.group_size

        # storage type, pack 8xint4 into int32
        params_dtype = torch.int32

        # Per-row scale columns
        def _n_scale_cols(in_features: int) -> int:
            return 1 if g == -1 else (in_features // g)

        # Register unpacked int4-as-int32 weights the loader will fill.
        w13 = torch.nn.Parameter(
            torch.empty(E, 2 * IN, H // self.packed_factor, dtype=params_dtype),
            requires_grad=False,
        )
        set_weight_attrs(w13, extra_weight_attrs)
        layer.register_parameter("w13_weight", w13)

        w2 = torch.nn.Parameter(
            torch.empty(E, H, IN // self.packed_factor, dtype=params_dtype),
            requires_grad=False,
        )
        set_weight_attrs(w2, extra_weight_attrs)
        layer.register_parameter("w2_weight", w2)

        # Register scales
        # KleidiAI groupwise kernels accepts float32 scales
        # KleidiAI groupwise kernels accepts bfloat16 scales
        scale_dtype = torch.float32 if g == -1 else torch.bfloat16

        w13_s = torch.nn.Parameter(
            torch.ones(E, 2 * IN, _n_scale_cols(H), dtype=scale_dtype),
            requires_grad=False,
        )
        set_weight_attrs(
            w13_s,
            {"quant_method": "channel" if g == -1 else "group", **extra_weight_attrs},
        )
        layer.register_parameter("w13_weight_scale", w13_s)

        w2_s = torch.nn.Parameter(
            torch.ones(E, H, _n_scale_cols(IN), dtype=scale_dtype), requires_grad=False
        )
        set_weight_attrs(
            w2_s,
            {"quant_method": "channel" if g == -1 else "group", **extra_weight_attrs},
        )
        layer.register_parameter("w2_weight_scale", w2_s)

        if self.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.zeros(E, 2 * IN, dtype=params_dtype), requires_grad=False
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)

            w2_bias = torch.nn.Parameter(
                torch.zeros(num_experts, hidden_size, dtype=params_dtype),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)

        # Placeholders for packed weights (will be replaced after packing)
        layer.register_parameter(
            "w13_weight_packed", torch.nn.Parameter(torch.empty(0), requires_grad=False)
        )
        set_weight_attrs(layer.w13_weight_packed, extra_weight_attrs)

        layer.register_parameter(
            "w2_weight_packed", torch.nn.Parameter(torch.empty(0), requires_grad=False)
        )
        set_weight_attrs(layer.w2_weight_packed, extra_weight_attrs)

        # dims for 4 bit fused matmuls
        layer.w13_in_features = H
        layer.w13_out_features = 2 * IN
        layer.w2_in_features = IN
        layer.w2_out_features = H
        layer.group_size = g

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # [expert, out_feature, groups] -> [expert, groupes, out_features]
        layer.w13_weight_scale = torch.nn.Parameter(
            layer.w13_weight_scale.transpose(1, 2).contiguous(),
            requires_grad=False,
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            layer.w2_weight_scale.transpose(1, 2).contiguous(),
            requires_grad=False,
        )

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return int4_w4a8_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w1_zp=None,
            w2_zp=None,
            block_shape=None,
        )

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # here we use Metax's `fused_experts`
        from vllm_metax.model_executor.layers.fused_moe.fused_moe import fused_experts

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


def int4_w4a8_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w1_zp: torch.Tensor | None,
    w2_zp: torch.Tensor | None,
    block_shape: list[int] | None = None,
) -> FusedMoEQuantConfig:
    """
    Construct a quant config for 16-bit float activations and int4 weights.
    """
    group_shape = GroupShape(*block_shape) if block_shape is not None else None
    return FusedMoEQuantConfig(
        _a1=FusedMoEQuantDesc("int8", shape=group_shape),
        _a2=FusedMoEQuantDesc("int8", shape=group_shape),
        _w1=FusedMoEQuantDesc("int4", group_shape, w1_scale, None, w1_zp),
        _w2=FusedMoEQuantDesc("int4", group_shape, w2_scale, None, w2_zp),
    )
