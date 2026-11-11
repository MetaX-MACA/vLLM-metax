# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from __future__ import annotations


from vllm.model_executor.layers.fused_moe import (
    RoutedExperts,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm_metax.registry.quant_config.mxfp4 import Mxfp4MoEMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    is_layer_skipped,
)

from vllm_metax.registry.quant_config.fp8 import MacaFp8Config

from vllm_metax.registry.quant_config._deepseek_v41_base import (
    DeepseekV4FP8Config as DeepseekV41FP8Config,
)
from vllm.model_executor.layers.quantization import register_quantization_config


@register_quantization_config("deepseek_v4_fp8")
class MacaDeepseekV41FP8Config(DeepseekV41FP8Config, MacaFp8Config):
    def get_quant_method(self, layer, prefix):
        if (
            isinstance(layer, LinearBase)
            and self.weight_block_size == [32, 32]
            and self.is_scale_e8m0
        ):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
                match_mode=self.ignored_layers_match_mode,
            ):
                return UnquantizedLinearMethod()

            from vllm_metax.registry.quant_config.fp8 import MctlassFp8LinearMethod

            return MctlassFp8LinearMethod(self)

        if isinstance(layer, RoutedExperts):
            if is_layer_skipped(
                prefix=prefix,
                ignored_layers=self.ignored_layers,
                fused_mapping=self.packed_modules_mapping,
            ):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            if self.expert_dtype == "fp4":
                if self.moe_quant_algo == "NVFP4":
                    raise NotImplementedError(
                        "NVFP4 quantization is not supported in vLLM-metax."
                    )
                return Mxfp4MoEMethod(layer.moe_config)
            # expert_dtype == "fp8": fall through to Fp8Config which
            # returns Fp8MoEMethod with block-wise float32 scales.
        return super().get_quant_method(layer, prefix)
