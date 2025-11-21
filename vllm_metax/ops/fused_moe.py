# SPDX-License-Identifier: Apache-2.0
# 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved. 
from vllm.model_executor.layers.fused_moe.layer import (
    UnquantizedFusedMoEMethod)

from vllm_metax.model_executor.layers.fused_moe.fused_moe import fused_experts


@UnquantizedFusedMoEMethod.register_oot
class MacaUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):

    def __init__(self, moe):
        super().__init__(moe)
        self.fused_experts = fused_experts  # type: ignore
