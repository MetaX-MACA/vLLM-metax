# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

from vllm.model_executor.layers.quantization.base_config import (  # noqa: E501
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    CompressedTensorsMoEMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
)

from vllm_metax.quant_config.compressed_tensors_moe import (
    MacaCompressedTensorsMoEMethod,
)

from vllm.model_executor.layers.quantization import register_quantization_config


@register_quantization_config("compressed-tensors")
class MacaCompressedTensorsConfig(CompressedTensorsConfig):
    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        origin_quant_method = super().get_quant_method(layer, prefix)
        if isinstance(origin_quant_method, CompressedTensorsMoEMethod):
            origin_quant_method = MacaCompressedTensorsMoEMethod.get_moe_method(
                self, layer, layer_name=prefix
            )
        return origin_quant_method
