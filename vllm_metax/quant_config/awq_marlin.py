# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

from typing import Optional, TYPE_CHECKING

from vllm.model_executor.layers.quantization.awq_marlin import AWQMarlinConfig

from vllm.model_executor.layers.quantization import register_quantization_config

if TYPE_CHECKING:
    from vllm.model_executor.layers.quantization import QuantizationMethods


@register_quantization_config("awq_marlin")
class MacaAWQMarlinConfig(AWQMarlinConfig):
    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant
    ) -> Optional["QuantizationMethods"]:
        return None
