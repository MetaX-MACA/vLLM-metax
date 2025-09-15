# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch
from vllm.model_executor.layers.fused_moe.layer import (FusedMoE)
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase)
from vllm.model_executor.layers.quantization.moe_wna16 import (
    MoeWNA16Config, is_layer_skipped_quant, MoeWNA16Method)

from vllm_metax.patch.model_executor.hook_register import (
    register_quantization_config)


# Remove configs of marlin
@register_quantization_config("moe_wna16")
class MacaMoeWNA16Config(MoeWNA16Config):
    """Config class for MOE WNA16 (W8A16/W4A16) quantization."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_marlin = False

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        if is_layer_skipped_quant(prefix, self.modules_to_not_convert):
            return UnquantizedLinearMethod()
        elif isinstance(layer, LinearBase):
            # Avoid circular import
            from vllm_metax.quant_config.awq import MacaAWQConfig
            from vllm_metax.quant_config.gptq import MacaGPTQConfig
            if self.linear_quant_method == "gptq":
                return MacaGPTQConfig.from_config(
                    self.full_config).get_quant_method(layer, prefix)
            elif self.linear_quant_method == "awq":
                return MacaAWQConfig.from_config(
                    self.full_config).get_quant_method(layer, prefix)
            else:
                raise ValueError("moe_wna16 only support gptq and awq.")
        elif isinstance(layer, FusedMoE):
            return MoeWNA16Method(self, layer.moe_config)
        return None
