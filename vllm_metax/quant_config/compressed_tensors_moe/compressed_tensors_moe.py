# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch
from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase
from vllm.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors_moe as vllm_ctm,
)


from vllm_metax.customized.layers.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)

from .compressed_tensors_moe_w8a8_int8 import CompressedTensorsW8A8Int8MoEMethod
from .compressed_tensors_moe_wna16 import CompressedTensorsWNA16MoEMethod
from .compressed_tensors_moe_w4a8_int4 import CompressedTensorsW4A8Int4MoEMethod


# -----------------------------------------------------------
# Note: We need to keep the method name **the same** as vLLM's
# -----------------------------------------------------------
class CompressedTensorsMoEMethod(vllm_ctm.CompressedTensorsMoEMethod):
    @staticmethod
    def get_moe_method(
        quant_config: "CompressedTensorsConfig",  # type: ignore # noqa E501
        layer: torch.nn.Module,
        layer_name: str,
    ) -> FusedMoEMethodBase:
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

        if scheme_dict is None:  # ignored layer
            return UnquantizedFusedMoEMethod(layer.moe_config)

        weight_quant = scheme_dict.get("weights")
        input_quant = scheme_dict.get("input_activations")

        # -------------------------------------------
        # Replace with Metax's MoE quantization methods by:
        #  - `weights_quant`
        #  - `input_quant`
        # -------------------------------------------
        origin_moe_method = None
        try:
            origin_moe_method = vllm_ctm.CompressedTensorsMoEMethod.get_moe_method(
                quant_config, layer, layer_name
            )
        except ValueError:
            # only handle CompressedTensorsW4A8Int4MoEMethod
            if not quant_config._is_dynamic_token_w4a8_int(weight_quant, input_quant):
                raise
        except Exception:
            raise

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
            vllm_ctm.logger.info_once(
                "Fallback to non-marlin CompressedTensorsWNA16MoEMethod"
            )
            return CompressedTensorsWNA16MoEMethod(
                weight_quant, input_quant, layer.moe_config
            )
        elif isinstance(origin_moe_method, vllm_ctm.CompressedTensorsW8A8Int8MoEMethod):
            return CompressedTensorsW8A8Int8MoEMethod(
                weight_quant, input_quant, layer.moe_config
            )
        elif quant_config._is_dynamic_token_w4a8_int(weight_quant, input_quant):
            # --------------------------------------------------------------------
            # Note!: On maca W4A8 is hardware supported. The quantization scheme
            #       is selected by `quant_config._is_dynamic_token_w4a8_int`. So we
            #       just need to re-implement and map with Int4MoEMethod here.
            # --------------------------------------------------------------------
            return CompressedTensorsW4A8Int4MoEMethod(
                weight_quant, input_quant, layer.moe_config
            )

        return origin_moe_method
