# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from vllm.model_executor.layers.fused_moe.layer import (
    UnquantizedFusedMoEMethod as vllm_UnquantizedFusedMoEMethod,
)

import torch

from vllm.platforms import current_platform, logger
import vllm_metax.envs as envs

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEActivationFormat,
    FusedMoEExpertsModular,
    FusedMoEPrepareAndFinalizeModular,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
)
from vllm.model_executor.layers.fused_moe.fused_batched_moe import BatchedTritonExperts

from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
)

from vllm_metax.model_executor.layers.fused_moe.fused_moe import (
    TritonExperts as mx_TritonExperts,
)

from vllm.model_executor.layers.fused_moe.fused_moe import (
    TritonExperts as vllm_TritonExperts,
)


def get_triton_experts_cls():
    if envs.USE_VLLM_TRITON_EXPERT:
        logger.info(
            "Using vLLM's fused MoE implementation for debugging and comparison."
        )
        return vllm_TritonExperts
    return mx_TritonExperts


TritonExperts = get_triton_experts_cls()


# -----------------------------------------------------------
# Note: We need to keep the method name **the same** as vLLM's
# -----------------------------------------------------------
@vllm_UnquantizedFusedMoEMethod.register_oot
class UnquantizedFusedMoEMethod(vllm_UnquantizedFusedMoEMethod):
    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        # -------------------------------------------------
        # Here in maca we use Triton for Modular MoE kernel
        self.unquantized_backend = UnquantizedMoeBackend.TRITON

    def select_gemm_impl(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalizeModular,
        layer: torch.nn.Module,
    ) -> FusedMoEExpertsModular:
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
            return TritonExperts(
                moe_config=self.moe,
                quant_config=self.moe_quant_config,
            )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        # Here in maca we choose `UnquantizedMoeBackend.TRITON` for kernel selection
        self.use_inplace = True
        self.kernel = mk.FusedMoEKernel(
            MoEPrepareAndFinalizeNoDPEPModular(),
            TritonExperts(
                moe_config=self.moe,
                quant_config=self.moe_quant_config,
            ),
            inplace=not self.moe.disable_inplace,
        )

    def forward_oot(
        self,
        layer: "FusedMoE",  # type: ignore[name-defined] # noqa: F821
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert self.kernel is not None

        return self.kernel.apply(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            shared_experts_input=shared_experts_input,
        )

    if current_platform.is_out_of_tree():
        forward_native = forward_oot
