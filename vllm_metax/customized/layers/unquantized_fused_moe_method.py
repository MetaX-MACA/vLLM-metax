# SPDX-License-Identifier: Apache-2.0
from vllm.model_executor.layers.fused_moe.layer import (
    UnquantizedFusedMoEMethod as vllm_UnquantizedFusedMoEMethod,
)

import torch

from vllm.platforms import current_platform, logger

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
    FusedMoEPermuteExpertsUnpermute,
    FusedMoEPrepareAndFinalize,
)

from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP,
)
from vllm_metax.model_executor.layers.fused_moe.fused_moe import (
    TritonExperts as mx_TritonExperts,
)
from vllm.model_executor.layers.fused_moe.fused_batched_moe import BatchedTritonExperts


# -----------------------------------------------------------
# Note: We need to keep the method name **the same** as vLLM's
# -----------------------------------------------------------
@vllm_UnquantizedFusedMoEMethod.register_oot
class UnquantizedFusedMoEMethod(vllm_UnquantizedFusedMoEMethod):
    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalize,
        layer: torch.nn.Module,
    ) -> FusedMoEPermuteExpertsUnpermute:
        assert self.moe_quant_config is not None
        if (
            prepare_finalize.activation_format
            == FusedMoEActivationFormat.BatchedExperts
        ):
            logger.debug("BatchedTritonExperts %s", self.moe)
            return BatchedTritonExperts(
                moe_config=self.moe,
                quant_config=self.moe_quant_config,
                max_num_tokens=self.moe.max_num_tokens,
                num_dispatchers=prepare_finalize.num_dispatchers(),
            )
        else:
            logger.debug("TritonExperts %s", self.moe)
            return mx_TritonExperts(
                moe_config=self.moe,
                quant_config=self.moe_quant_config,
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        # Here in maca we choose `UnquantizedMoeBackend.TRITON` for kernel selection
        self.use_inplace = True
        self.kernel = mk.FusedMoEModularKernel(
            MoEPrepareAndFinalizeNoEP(),
            mx_TritonExperts(
                moe_config=self.moe,
                quant_config=self.moe_quant_config,
            ),
        )

    def forward_oot(
        self,
        layer: "FusedMoE",  # type: ignore[name-defined] # noqa: F821
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.kernel(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=self.use_inplace,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
        )

    if current_platform.is_out_of_tree():
        forward_native = forward_oot
