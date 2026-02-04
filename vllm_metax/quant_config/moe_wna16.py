# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Optional

import torch
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.moe_wna16 import (
    MoeWNA16Config,
    is_layer_skipped_quant,
    MoeWNA16Method as vllm_MoeWNA16Method,
)
from vllm.model_executor.layers.fused_moe.unquantized_fused_moe_method import (
    UnquantizedFusedMoEMethod,
)

from vllm.model_executor.layers.quantization import register_quantization_config


# Remove configs of marlin
@register_quantization_config("moe_wna16")
class MacaMoeWNA16Config(MoeWNA16Config):
    """Config class for MOE WNA16 (W8A16/W4A16) quantization."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.use_marlin = False

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        if is_layer_skipped_quant(prefix, self.modules_to_not_convert):
            if isinstance(layer, FusedMoE):
                return UnquantizedFusedMoEMethod(layer.moe_config)
            return UnquantizedLinearMethod()
        elif isinstance(layer, LinearBase):
            # Avoid circular import
            from vllm_metax.quant_config.awq import MacaAWQConfig
            from vllm_metax.quant_config.gptq import MacaGPTQConfig

            if self.linear_quant_method == "gptq":
                return MacaGPTQConfig.from_config(self.full_config).get_quant_method(
                    layer, prefix
                )
            elif self.linear_quant_method == "awq":
                return MacaAWQConfig.from_config(self.full_config).get_quant_method(
                    layer, prefix
                )
            else:
                raise ValueError("moe_wna16 only support gptq and awq.")
        elif isinstance(layer, FusedMoE):
            return MoeWNA16Method(self, layer.moe_config)
        return None


# -----------------------------------------------------------
# Note: We need to keep the method name **the same** as vLLM's
# -----------------------------------------------------------
class MoeWNA16Method(vllm_MoeWNA16Method):
    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # here we use plugin's `fused_experts`
        from vllm_metax.model_executor.layers.fused_moe.fused_moe import fused_experts

        assert layer.activation == "silu", "Only SiLU activation is supported."

        return fused_experts(
            x,
            layer.w13_qweight,
            layer.w2_qweight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=True,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            quant_config=self.moe_quant_config,
        )
