# SPDX-License-Identifier: Apache-2.0
from vllm_metax.model_executor.layers.fused_moe.fused_moe import metax_fused_experts
from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod


@UnquantizedFusedMoEMethod.register_oot
class MetaxUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    def __init__(self, moe):
        super().__init__(moe)
        self.fused_experts = metax_fused_experts  # type: ignore