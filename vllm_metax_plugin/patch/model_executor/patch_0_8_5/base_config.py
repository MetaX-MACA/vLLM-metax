# SPDX-License-Identifier: Apache-2.0

import vllm
from vllm_metax_plugin.patch.hook_registry import register_patch
from vllm.logger import init_logger

logger = init_logger(__name__)

from typing import Optional

def override_quantization_method(cls, hf_quant_cfg,
                                    user_quant) -> Optional[str]:
    """
        Detects if this quantization method can support a given checkpoint
        format by overriding the user specified quantization method -- 
        this method should only be overwritten by subclasses in exceptional 
        circumstances
    """
    if(user_quant != None):
        return user_quant
    else:
        return hf_quant_cfg["quant_method"]
    
vllm.model_executor.layers.quantization.base_config.QuantizationConfig.override_quantization_method = classmethod(override_quantization_method)
register_patch("vllm.model_executor.layers.quantization.base_config", "QuantizationConfig.override_quantization_method", classmethod(override_quantization_method))