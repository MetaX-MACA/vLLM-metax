# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch
from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe import (
    CompressedTensorsMoEMethod as vllm_ct_moe_method,
    logger,
)

from compressed_tensors.quantization import (
    ActivationOrdering,
    QuantizationStrategy,
)

from compressed_tensors import CompressionFormat
from vllm_metax.customized.layers.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)

from .compressed_tensors_moe_w8a8_int8 import (
    CompressedTensorsW8A8Int8MoEMethod,
)


from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (  # noqa
    WNA16_SUPPORTED_BITS,
)

from .compressed_tensors_moe_w4a8_int4 import CompressedTensorsW4A8Int4MoEMethod


# -----------------------------------------------------------
# Note: We need to keep the method name **the same** as vLLM's
# -----------------------------------------------------------
class CompressedTensorsMoEMethod(vllm_ct_moe_method):
    @staticmethod
    def get_moe_method(
        quant_config: "CompressedTensorsConfig",  # type: ignore # noqa E501
        layer: torch.nn.Module,
        layer_name: str,
    ) -> FusedMoEMethodBase:
        # FusedMoE was made by combining multiple Linears so need to
        # make sure quantization config for Linear can target it
        quant_config._add_fused_moe_to_target_scheme_map()
        unfused_names = [
            layer_name + proj_name
            for proj_name in [".0.gate_proj", ".0.up_proj", ".0.down_proj"]
        ]
        # TODO: refactor this to use expert_mapping and check all layer numbers
        all_scheme_dicts = [
            quant_config.get_scheme_dict(layer, name) for name in unfused_names
        ]
        scheme_dict = all_scheme_dicts.pop()

        # multiple schemes found
        if not all([cur_dict == scheme_dict for cur_dict in all_scheme_dicts]):
            raise ValueError(
                "All MoE projections need to have same "
                "quantization scheme but found multiple"
            )

        if scheme_dict is None:  # ignored layer
            return UnquantizedFusedMoEMethod(layer.moe_config)

        # TODO: @dsikka: refactor this to use schemes as other kernels
        # are supported + check if the layer is being ignored.
        weight_quant = scheme_dict.get("weights")
        input_quant = scheme_dict.get("input_activations")
        format = scheme_dict.get("format")

        if quant_config._is_wNa16_group_channel(weight_quant, input_quant):
            # group_size=None means channelwise
            group_size = weight_quant.group_size or -1  # noqa: F841

            valid_format_and_bits = (
                weight_quant.num_bits in WNA16_SUPPORTED_BITS
                and format == CompressionFormat.pack_quantized.value
            )

            if not valid_format_and_bits:
                raise ValueError(
                    "For Fused MoE layers, only format: ",
                    f"{CompressionFormat.pack_quantized.value} ",
                    f" and bits: {WNA16_SUPPORTED_BITS} is supported ",
                    f"but got format: {CompressionFormat.pack_quantized.value} "
                    f" and bits: {weight_quant.num_bits}",
                )

            from .compressed_tensors_moe_wna16 import (
                CompressedTensorsWNA16MoEMethod,
            )

            if (
                weight_quant.strategy == QuantizationStrategy.GROUP
                and weight_quant.actorder
                in (ActivationOrdering.GROUP, ActivationOrdering.DYNAMIC)
            ):
                raise ValueError(
                    "WNA16MoE is not supported with actorder=group/dynamic."
                )
            logger.info_once("Using CompressedTensorsWNA16MoEMethod")
            return CompressedTensorsWNA16MoEMethod(
                weight_quant, input_quant, layer.moe_config
            )

        if quant_config._is_dynamic_token_w8a8(weight_quant, input_quant):
            return CompressedTensorsW8A8Int8MoEMethod(
                weight_quant, input_quant, layer.moe_config
            )

        if quant_config._is_dynamic_token_w4a8_int(weight_quant, input_quant):
            # --------------------------------------------------------------------
            # Note!: On maca W4A8 is hardware supported. The quantization scheme
            #       is selected by `quant_config._is_dynamic_token_w4a8_int`. So we
            #       just need to re-implement and map with Int4MoEMethod here.
            # --------------------------------------------------------------------
            return CompressedTensorsW4A8Int4MoEMethod(
                weight_quant, input_quant, layer.moe_config
            )
