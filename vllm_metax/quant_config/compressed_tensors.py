
# SPDX-License-Identifier: Apache-2.0
from typing import Optional
import torch
from vllm.model_executor.layers.quantization.base_config import (  # noqa: E501
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors as vllm_ct,
)
from vllm_metax.quant_config.compressed_tensors_moe import CompressedTensorsMoEMethod as MacaCompressedTensorsMoEMethod
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.attention.layer import Attention
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.transform.linear import (  # noqa: E501
    CompressedTensorsLinearTransformMethod,
    get_linear_transform_schemes,
)
from vllm.model_executor.layers.fused_moe import FusedMoE

@register_quantization_config("compressed-tensors")
class MacaCompressedTensorsConfig(vllm_ct.CompressedTensorsConfig):
    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        # Replace with Metax's MoE quantization methods
        if isinstance(layer, LinearBase):
            # collect schemes
            quant_scheme = self.get_scheme(layer=layer, layer_name=prefix)
            input_tfms, output_tfms = get_linear_transform_schemes(
                layer, prefix, self.transform_config, self.packed_modules_mapping
            )

            # choose quantization method
            quant_method: LinearMethodBase = UnquantizedLinearMethod()
            if quant_scheme is not None:
                layer.scheme = quant_scheme
                quant_method = vllm_ct.CompressedTensorsLinearMethod(self)
            
            # choose transform method
            if any((input_tfms, output_tfms)):
                return CompressedTensorsLinearTransformMethod.from_schemes(
                    quant_method, quant_scheme, input_tfms, output_tfms
                )
            else:
                return quant_method

        if isinstance(layer, Attention):
            return vllm_ct.CompressedTensorsKVCacheMethod(self)
        if isinstance(layer, FusedMoE):
            return MacaCompressedTensorsMoEMethod.get_moe_method(
                self, layer, layer_name=prefix
            )
            
        return None
