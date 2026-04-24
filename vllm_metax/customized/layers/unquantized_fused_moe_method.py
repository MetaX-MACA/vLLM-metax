# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from vllm.model_executor.layers.fused_moe.layer import (
    UnquantizedFusedMoEMethod as vllm_UnquantizedFusedMoEMethod,
)

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
)

from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    UnquantizedMoeBackend,
)

from vllm_metax.utils.fused_moe import get_triton_experts_cls


def backend_to_kernel_cls(
    backend: UnquantizedMoeBackend,
) -> type[mk.FusedMoEExperts]:
    if backend == UnquantizedMoeBackend.TRITON:
        TritonExperts = get_triton_experts_cls()
        return TritonExperts

    elif backend == UnquantizedMoeBackend.BATCHED_TRITON:
        from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
            BatchedTritonExperts,
        )

        return BatchedTritonExperts


def select_unquantized_moe_backend(
    moe_config: FusedMoEConfig,
) -> tuple[UnquantizedMoeBackend, type[mk.FusedMoEExperts] | None]:
    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if moe_config.moe_parallel_config.use_batched_activation_format
        else mk.FusedMoEActivationFormat.Standard
    )
    requested_backend = UnquantizedMoeBackend.TRITON
    if (
        activation_format == mk.FusedMoEActivationFormat.BatchedExperts
        and requested_backend == UnquantizedMoeBackend.TRITON
    ):
        requested_backend = UnquantizedMoeBackend.BATCHED_TRITON

    kernel_cls = backend_to_kernel_cls(requested_backend)
    return requested_backend, kernel_cls


# -----------------------------------------------------------
# Note: We need to keep the method name **the same** as vLLM's
# -----------------------------------------------------------
@vllm_UnquantizedFusedMoEMethod.register_oot
class UnquantizedFusedMoEMethod(vllm_UnquantizedFusedMoEMethod):
    def __init__(self, moe: FusedMoEConfig):
        super(vllm_UnquantizedFusedMoEMethod, self).__init__(moe)
        # -------------------------------------------------
        # Here in maca we use Triton for Modular MoE kernel
        moe.moe_backend = "triton"
        self.unquantized_backend, self.experts_cls = select_unquantized_moe_backend(
            moe_config=self.moe,
        )

    def forward_oot(
        self,
        layer: "FusedMoE",  # type: ignore[name-defined] # noqa: F821
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
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
