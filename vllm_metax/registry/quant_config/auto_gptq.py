# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

import torch
from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig
from vllm.model_executor.layers.quantization.auto_gptq import (
    AutoGPTQLinearMethod as vllm_AutoGPTQLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.utils.gptq_utils import (
    get_linear_quant_method,
)

from vllm_metax.customized.kernels import (
    mp_linear_kernel,  # noqa: F401
)


@register_quantization_config("auto_gptq")
@register_quantization_config("gptq")
class MacaAutoGPTQConfig(AutoGPTQConfig):
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16]

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase| None":
        if isinstance(layer, RoutedExperts):
            # GPTQ MoE support: fall back to MoeWNA16 for broad compatibility
            from vllm_metax.quant_config.moe_wna16 import MacaMoeWNA16Config

            return MacaMoeWNA16Config.from_config(self.full_config).get_quant_method(
                layer, prefix
            )

        quant_method = get_linear_quant_method(
            self, layer, prefix, AutoGPTQLinearMethod
        )
        if quant_method is None:
            return None
        return quant_method


# -----------------------------------------------------------
# Note: We need to keep the method name **the same** as vLLM's
# -----------------------------------------------------------
class AutoGPTQLinearMethod(vllm_AutoGPTQLinearMethod):
    pass
