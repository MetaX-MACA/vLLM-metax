# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)

from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationStrategy,
)

from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w8a8_int8 import (
    CompressedTensorsW8A8Int8MoEMethod as vllm_ctm_w8a8_int8,
)

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
)

from vllm.model_executor.utils import replace_parameter

from vllm_metax.model_executor.layers.fused_moe.oracle.int8 import (
    convert_to_int8_moe_kernel_format,
    make_int8_moe_kernel,
    make_int8_moe_quant_config,
    select_int8_moe_backend,
)


# -----------------------------------------------------------
# Note: We need to keep the method name **the same** as vLLM's
# -----------------------------------------------------------
class CompressedTensorsW8A8Int8MoEMethod(vllm_ctm_w8a8_int8):
    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
    ):
        super(vllm_ctm_w8a8_int8, self).__init__(moe)
        self.weight_quant = weight_quant
        self.input_quant = input_quant

        per_channel = (
            self.weight_quant.strategy == QuantizationStrategy.CHANNEL
            and self.input_quant.strategy == QuantizationStrategy.TOKEN
        )
        if not per_channel:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found "
                f"{self.weight_quant}, {self.input_quant}"
            )

        self.static_input_scales = not self.input_quant.dynamic
        if self.static_input_scales:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found static input scales."
            )

        # Select Int8 MoE backend.
        self.int8_backend, self.experts_cls = select_int8_moe_backend(
            config=self.moe,
            weight_key=kInt8StaticChannelSym,
            activation_key=kInt8DynamicTokenSym,
        )

    @property
    def supports_eplb(self) -> bool:
        # EPLB remaps expert IDs and rearranges expert weights; the local INT8
        # kernels consume the same per-channel/per-token layout.
        return True

    def process_weights_after_loading(self, layer: RoutedExperts) -> None:
        w13, w2 = convert_to_int8_moe_kernel_format(
            int8_backend=self.int8_backend,
            w13=layer.w13_weight,
            w2=layer.w2_weight,
            layer=layer,
            w13_scale=layer.w13_weight_scale,
        )
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)

        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        assert self.experts_cls is not None
        self.moe_kernel = make_int8_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
            routing_tables=layer._expert_routing_tables(),
        )
        self.moe_kernel.fused_experts.process_weights_after_loading(layer)

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig:
        return make_int8_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            w1_bias=getattr(layer, "w13_bias", None),
            w2_bias=getattr(layer, "w2_bias", None),
            per_act_token_quant=True,
            layer=layer,
        )
