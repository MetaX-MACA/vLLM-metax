# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

import torch

from vllm.model_executor.layers.quantization.base_config import (  # noqa: E501
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.compressed_tensors import (
    compressed_tensors as vllm_ct,
)
from vllm_metax.quant_config.compressed_tensors_moe.compressed_tensors_moe import (
    CompressedTensorsMoEMethod,
)

from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.fused_moe import FusedMoE


@register_quantization_config("compressed-tensors")
class MacaCompressedTensorsConfig(vllm_ct.CompressedTensorsConfig):
    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> "QuantizeMethodBase | None":
        try:
            origin_quant_method = super().get_quant_method(layer, prefix)
        except ValueError:
            # Note: w4a8 may trigger ValueError in the CompressedTensorsMoEMethod,
            # but we'd handle it in our custom method below.
            # So we catch the exception and ensure it's a FusedMoE layer.
            if not isinstance(layer, FusedMoE):
                raise
        except Exception:
            raise

        # Replace with Metax's MoE quantization methods
        if isinstance(layer, FusedMoE):
            return CompressedTensorsMoEMethod.get_moe_method(
                self, layer, layer_name=prefix
            )

        return origin_quant_method
