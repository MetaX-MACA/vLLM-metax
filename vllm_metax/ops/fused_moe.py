# SPDX-License-Identifier: Apache-2.0
from vllm_metax.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod


@UnquantizedFusedMoEMethod.register_oot
class MacaUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    def __init__(self, moe):
        super().__init__(moe)
        self.fused_experts = fused_experts  # type: ignore