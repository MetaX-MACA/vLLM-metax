# SPDX-License-Identifier: Apache-2.0
# ------------------------------------------------------------------------
# Note: This file is a patch to opt dp all2all
# ------------------------------------------------------------------------
import torch
import vllm.envs as envs
from vllm_metax import envs as mx_envs

from vllm.distributed import tensor_model_parallel_all_reduce

from vllm.model_executor.layers.fused_moe.runner.default_moe_runner import (
    DefaultMoERunner,
)


@property
def use_combine_allreduce(self):
    return (
        self.moe_config.moe_parallel_config.dp_size > 1
        and mx_envs.MACA_DP_OPT
        and (
            envs.VLLM_ALL2ALL_BACKEND == "naive"
            or envs.VLLM_ALL2ALL_BACKEND == "allgather_reducescatter"
        )
    )


def must_reduce_shared_expert_outputs(self) -> bool:
    assert self.quant_method is not None
    return (
        self.quant_method.moe_kernel is not None
        and self.quant_method.moe_kernel.output_is_reduced()
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


DefaultMoERunner.use_combine_allreduce = use_combine_allreduce
DefaultMoERunner.must_reduce_shared_expert_outputs = must_reduce_shared_expert_outputs
DefaultMoERunner.maybe_all_reduce_tensor_model_parallel = (
    maybe_all_reduce_tensor_model_parallel
)
