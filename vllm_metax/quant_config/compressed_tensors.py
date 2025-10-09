# SPDX-License-Identifier: Apache-2.0

# TODO: hotfix for subprocess while unpickle, remove after torch2.8 is released
from vllm_metax.patch.hotfix import patch_utils  # noqa: F401

from typing import Optional

import torch

from vllm.model_executor.layers.quantization.base_config import (  # noqa: E501
    QuantizeMethodBase)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe import (  # noqa: E501
    CompressedTensorsMoEMethod)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig)

from vllm_metax.quant_config.compressed_tensors_moe import MacaCompressedTensorsMoEMethod

from vllm_metax.patch.model_executor.hook_register import register_quantization_config


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
                self, layer)
        return origin_quant_method
