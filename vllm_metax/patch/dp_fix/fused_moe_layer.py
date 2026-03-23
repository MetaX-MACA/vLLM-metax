# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# ------------------------------------------------------------------------
# Note: This file is a patch to opt dp all2all
# ------------------------------------------------------------------------
import torch
import vllm.envs as envs
from vllm_metax import envs as mx_envs

from vllm.model_executor.layers.fused_moe.fused_moe_modular_method import (
    FusedMoEModularMethod,
)
from vllm.distributed import tensor_model_parallel_all_reduce

from vllm.model_executor.layers.fused_moe.layer import FusedMoE


class MacaFusedMoE(FusedMoE):
    @property
    def use_combine_allreduce(self):
        return (
            self.moe_parallel_config.dp_size > 1
            and mx_envs.MACA_DP_OPT
            and (
                envs.VLLM_ALL2ALL_BACKEND == "naive"
                or envs.VLLM_ALL2ALL_BACKEND == "allgather_reducescatter"
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
