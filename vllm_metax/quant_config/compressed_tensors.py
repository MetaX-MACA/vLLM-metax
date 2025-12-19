# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

from vllm.model_executor.layers.quantization.base_config import (  # noqa: E501
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors_moe as vllm_ctm,
)
from vllm.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors as vllm_ct,
)

from vllm_metax.quant_config.compressed_tensors_moe import CompressedTensorsMoEMethod

from vllm.model_executor.layers.quantization import register_quantization_config

# -----------------------------------------------------------
# Note: We need to keep the name **consistent** with vLLM's
# -----------------------------------------------------------


@register_quantization_config("compressed-tensors")
class MacaCompressedTensorsConfig(vllm_ct.CompressedTensorsConfig):
    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        origin_quant_method = super().get_quant_method(layer, prefix)
        # Replace with Metax's MoE quantization methods
        if isinstance(origin_quant_method, vllm_ctm.CompressedTensorsMoEMethod):
            origin_quant_method = CompressedTensorsMoEMethod.get_moe_method(
                self, layer, layer_name=prefix
            )
        return origin_quant_method
