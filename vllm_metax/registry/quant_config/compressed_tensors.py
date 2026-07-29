# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

import torch

from vllm.model_executor.layers.quantization.base_config import (  # noqa: E501
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsLinearMethod,
    CompressedTensorsKVCacheMethod,
)
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationType,
)
from vllm_metax.quant_config.compressed_tensors_moe.compressed_tensors_moe import (
    CompressedTensorsMoEMethod,
)

from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.compressed_tensors.transform.linear import (  # noqa: E501
    CompressedTensorsLinearTransformMethod,
    get_linear_transform_schemes,
)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_embedding import (  # noqa: E501
    CompressedTensorsEmbeddingWNA16Int,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)


@register_quantization_config("compressed-tensors")
class MacaCompressedTensorsConfig(CompressedTensorsConfig):
    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> "QuantizeMethodBase | None":
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
                quant_method = CompressedTensorsLinearMethod(self)

            # choose transform method
            if any((input_tfms, output_tfms)):
                return CompressedTensorsLinearTransformMethod.from_schemes(
                    quant_method, quant_scheme, input_tfms, output_tfms
                )

            else:
                return quant_method

        if isinstance(layer, ParallelLMHead):
            try:
                quant_scheme = self.get_scheme(layer=layer, layer_name=prefix)
            except ValueError:
                quant_scheme = None
            if quant_scheme is not None:
                layer.scheme = quant_scheme
                return CompressedTensorsLinearMethod(self)

        # ParallelLMHead subclasses VocabParallelEmbedding but is handled above as
        # a linear; only true embedding lookups land here.
        if isinstance(layer, VocabParallelEmbedding):
            scheme_dict = self.get_scheme_dict(layer, layer_name=prefix)
            weight_quant = scheme_dict.get("weights") if scheme_dict else None
            if weight_quant is None:
                return None  # unquantized embedding
            if not (
                isinstance(weight_quant, QuantizationArgs)
                and self._is_wNa16_group_channel(weight_quant, None)
                and weight_quant.type == QuantizationType.INT
            ):
                raise ValueError(
                    "compressed-tensors embeddings only support weight-only INT "
                    f"group/channel (WNA16) quantization, got: {weight_quant}"
                )
            return CompressedTensorsEmbeddingWNA16Int(weight_quant)

        if isinstance(layer, Attention):
            return CompressedTensorsKVCacheMethod(self)
        if isinstance(layer, RoutedExperts):
            return CompressedTensorsMoEMethod.get_moe_method(
                self, layer, layer_name=prefix
            )
        return None
