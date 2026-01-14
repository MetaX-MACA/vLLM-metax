# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# ------------------------------------------------------------------------
# Note: This file is a patch to opt dp all2all
# ------------------------------------------------------------------------
import torch
import vllm.envs as envs
from vllm_metax import envs as mx_envs

from vllm._aiter_ops import rocm_aiter_ops
from collections.abc import Callable, Iterable

from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import (
    FusedMoEModularMethod,
)
from vllm.distributed import tensor_model_parallel_all_reduce

from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm_metax.model_executor.layers.fused_moe.fused_moe import grouped_topk as mx_grouped_topk

class MacaFusedMoE(FusedMoE):
    
    @property
    def use_combine_allreduce(self):
        return self.moe_parallel_config.dp_size > 1 and mx_envs.MACA_DP_OPT \
            and (envs.VLLM_ALL2ALL_BACKEND == "naive" \
                or envs.VLLM_ALL2ALL_BACKEND == "allgather_reducescatter")
    
    def must_reduce_shared_expert_outputs(self) -> bool:
        assert self.quant_method is not None
        return (
            isinstance(self.quant_method, FusedMoEModularMethod)
            and self.quant_method.fused_experts.output_is_reduced()
            or self.use_combine_allreduce
        )

    def maybe_all_reduce_tensor_model_parallel(self, final_hidden_states: torch.Tensor):
        """
        Some combine kernels reduce across GPU ranks by default.
        """
        if self.must_reduce_shared_expert_outputs():
            return final_hidden_states
        else:
            return tensor_model_parallel_all_reduce(final_hidden_states)
    
    @staticmethod
    def select_experts(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        use_grouped_topk: bool,
        renormalize: bool,
        topk_group: int | None = None,
        num_expert_group: int | None = None,
        custom_routing_function: Callable | None = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: torch.Tensor | None = None,
        indices_type: torch.dtype | None = None,
        enable_eplb: bool = False,
        expert_map: torch.Tensor | None = None,
        expert_load_view: torch.Tensor | None = None,
        logical_to_physical_map: torch.Tensor | None = None,
        logical_replica_count: torch.Tensor | None = None,
        global_num_experts: int | None = None,
        zero_expert_num: int | None = None,
        zero_expert_type: str | None = None,
        num_fused_shared_experts: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route the input hidden states to the top-k experts based on the
        router logits.

        Returns:
                (topk_weights, topk_ids, zero_expert_result)
                (tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
                The weights, expert ids, and zero expert computation result.

            **Compatibility**: When EPLB is not enabled, the returned ids are
            equivalent to global logical ids, so should be compatible with
            plain MoE implementations without redundant experts.
        """
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            fused_topk,
            fused_topk_bias,
        )

        # Check if we should use a routing simulation strategy
        routing_strategy = envs.VLLM_MOE_ROUTING_SIMULATION_STRATEGY
        if routing_strategy != "":
            topk_weights, topk_ids = RoutingSimulator.simulate_routing(
                hidden_states=hidden_states,
                router_logits=router_logits,
                strategy_name=routing_strategy,
                top_k=top_k,
                indices_type=indices_type,
            )

        # DeepSeekv2 uses grouped_top_k
        elif use_grouped_topk:
            assert topk_group is not None
            assert num_expert_group is not None
            if rocm_aiter_ops.is_fused_moe_enabled():
                if not rocm_aiter_ops.is_fusion_moe_shared_experts_enabled():
                    assert num_fused_shared_experts == 0
                grouped_topk_impl = partial(
                    rocm_aiter_grouped_topk,
                    num_fused_shared_experts=num_fused_shared_experts,
                )
            else:
                # ┌-------------------------------  Metax Modification --------------------------------┐
                grouped_topk_impl = mx_grouped_topk
                # └-------------------------------- Metax Modification --------------------------------┘

            topk_weights, topk_ids = grouped_topk_impl(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                scoring_func=scoring_func,
                routed_scaling_factor=routed_scaling_factor,
                e_score_correction_bias=e_score_correction_bias,
            )
            if indices_type is not None:
                topk_ids = topk_ids.to(dtype=indices_type)
        elif e_score_correction_bias is not None:
            topk_weights, topk_ids = fused_topk_bias(
                hidden_states=hidden_states,
                gating_output=router_logits,
                e_score_correction_bias=e_score_correction_bias.data,
                topk=top_k,
                renormalize=renormalize,
            )
            if routed_scaling_factor is not None:
                topk_weights *= routed_scaling_factor
        elif custom_routing_function is None:
            topk_weights, topk_ids, token_expert_indices = fused_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                indices_type=indices_type,
            )
        else:
            topk_weights, topk_ids = custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
            )
            if indices_type is not None:
                topk_ids = topk_ids.to(dtype=indices_type)

        if enable_eplb:
            assert expert_load_view is not None
            assert logical_to_physical_map is not None
            assert logical_replica_count is not None

            topk_ids = eplb_map_to_physical_and_record(
                topk_ids=topk_ids,
                expert_load_view=expert_load_view,
                logical_to_physical_map=logical_to_physical_map,
                logical_replica_count=logical_replica_count,
                indices_type=indices_type,
            )

        assert topk_ids.dtype == indices_type or indices_type is None

        # Compute zero expert result if needed
        if (
            zero_expert_num is not None
            and zero_expert_num > 0
            and zero_expert_type is not None
            and global_num_experts is not None
        ):
            zero_expert_result = zero_experts_compute_triton(
                expert_indices=topk_ids,
                expert_scales=topk_weights,
                num_experts=global_num_experts,
                zero_expert_type=zero_expert_type,
                hidden_states=hidden_states,
            )
        else:
            zero_expert_result = None
        return topk_weights, topk_ids, zero_expert_result

FusedMoE.select_experts = MacaFusedMoE.select_experts
FusedMoE.use_combine_allreduce = MacaFusedMoE.use_combine_allreduce
FusedMoE.must_reduce_shared_expert_outputs = MacaFusedMoE.must_reduce_shared_expert_outputs
FusedMoE.maybe_all_reduce_tensor_model_parallel = MacaFusedMoE.maybe_all_reduce_tensor_model_parallel