# SPDX-License-Identifier: Apache-2.0

from pydantic import BaseModel
from vllm.model_executor.layers.fused_moe import FusedMoE
from compressed_tensors.quantization import QuantizationType
from compressed_tensors.config import SparsityCompressionConfig
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.compressed_tensors.utils import should_ignore_layer
from vllm.model_executor.layers.linear import (LinearBase,  UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (CompressedTensorsKVCacheMethod,
                                                                                           CompressedTensorsLinearMethod)
import torch
from typing import Optional, Any, List, Dict
from .compressed_tensors_moe import MetaxCompressedTensorsMoEMethod

from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig

class MetaxCompressedTensorsConfig(CompressedTensorsConfig):

    def __init__(
        self,
        target_scheme_map: Dict[str, Any],
        ignore: List[str],
        quant_format: str,
        sparsity_scheme_map: Dict[str, SparsityCompressionConfig],
        sparsity_ignore_list: List[str],
        kv_cache_scheme: Optional[Dict[str, Any]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            target_scheme_map=target_scheme_map,
            ignore=ignore,
            quant_format=quant_format,
            sparsity_scheme_map=sparsity_scheme_map,
            sparsity_ignore_list=sparsity_ignore_list,
            kv_cache_scheme=kv_cache_scheme,
            config=config,
        )

    @classmethod
    def from_config_instance(cls, cfg: CompressedTensorsConfig):
        return cls(
            target_scheme_map=cfg.target_scheme_map,
            ignore=cfg.ignore,
            quant_format=cfg.quant_format,
            sparsity_scheme_map=cfg.sparsity_scheme_map,
            sparsity_ignore_list=cfg.sparsity_ignore_list,
            kv_cache_scheme=cfg.kv_cache_scheme,
            config=cfg.config,
        )
    

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> Optional["QuantizeMethodBase"]:
        from vllm.attention.layer import Attention  # Avoid circular import

        # Check if the layer is skipped for quantization.
        # TODO (@robertgshaw2): support module names
        if should_ignore_layer(prefix,
                               ignore=self.ignore,
                               fused_mapping=self.packed_modules_mapping):
            return UnquantizedLinearMethod()
        if isinstance(layer, LinearBase):
            scheme = self.get_scheme(layer=layer, layer_name=prefix)
            if scheme is None:
                return UnquantizedLinearMethod()
            layer.scheme = scheme
            return CompressedTensorsLinearMethod(self)
        if isinstance(layer, Attention):
            return CompressedTensorsKVCacheMethod(self)
        if isinstance(layer, FusedMoE):
            return MetaxCompressedTensorsMoEMethod.get_moe_method(self, layer)
        return None

    def _check_scheme_supported(self,
                            min_capability: int,
                            error: bool = True,
                            match_exact: bool = False) -> bool:
        return False

    def _is_dynamic_token_int8_w8a8(self, weight_quant: BaseModel,
                        input_quant: BaseModel) -> bool:
        # Confirm weights and activations quantized.
        if weight_quant is None or input_quant is None:
            return False
            
        is_int8_w8a8 = (weight_quant.type == QuantizationType.INT and input_quant.type == QuantizationType.INT)
        return is_int8_w8a8 and self._is_dynamic_token_w8a8(weight_quant, input_quant)  
