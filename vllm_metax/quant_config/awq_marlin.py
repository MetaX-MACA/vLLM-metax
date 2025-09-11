# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from vllm.model_executor.layers.quantization.awq_marlin import (
    AWQMarlinConfig, QuantizationMethods)

from vllm_metax.patch.model_executor.patch.hook_register import (
    register_quantization_config)


@register_quantization_config("awq_marlin")
class MacaAWQMarlinConfig(AWQMarlinConfig):

    @classmethod
    def override_quantization_method(
            cls, hf_quant_cfg, user_quant) -> Optional[QuantizationMethods]:
        return None
