# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# ------------------------------------------------------------------------
# Note: This file is a patch to opt dp all2all
# ------------------------------------------------------------------------
import torch
from vllm_metax import envs as mx_envs

from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import (
    FusedMoEModularMethod,
)
from vllm.distributed import tensor_model_parallel_all_reduce

from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm_metax.model_executor.layers.fused_moe.fused_moe import (
    grouped_topk as mx_grouped_topk,
)
from vllm.model_executor.layers.fused_moe.fused_moe import GroupedTopk


class MacaFusedMoE(FusedMoE):
    @property
    def use_combine_allreduce(self):
        return (
            self.moe_parallel_config.dp_size > 1
            and mx_envs.MACA_DP_OPT
            and (
                self.moe_parallel_config.all2all_backend == "naive"
                or self.moe_parallel_config.all2all_backend == "allgather_reducescatter"
            )
        )

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


FusedMoE.use_combine_allreduce = MacaFusedMoE.use_combine_allreduce
FusedMoE.must_reduce_shared_expert_outputs = (
    MacaFusedMoE.must_reduce_shared_expert_outputs
)
FusedMoE.maybe_all_reduce_tensor_model_parallel = (
    MacaFusedMoE.maybe_all_reduce_tensor_model_parallel
)


@GroupedTopk.register_oot
class MacaGroupedTopk(GroupedTopk):
    def forward_oot(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        e_score_correction_bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return mx_grouped_topk(
            hidden_states,
            gating_output,
            self.topk,
            self.renormalize,
            self.num_expert_group,
            self.topk_group,
            self.scoring_func,
            self.routed_scaling_factor,
            e_score_correction_bias,
        )


GroupedTopk.forward_native = MacaGroupedTopk.forward_oot
