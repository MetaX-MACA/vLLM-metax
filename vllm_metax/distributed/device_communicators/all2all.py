# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch

from vllm.distributed import get_dp_group, get_ep_group, get_tp_group
from vllm.forward_context import get_forward_context

from vllm.distributed.device_communicators.base_device_communicator import All2AllManagerBase

class CoArAll2AllManager(All2AllManagerBase):
    """
    A opt implementation of all2all communication.
    It uses all-reduce under the hood, which is not
    efficient at all. The main purpose is for testing and
    debugging.
    """

    def __init__(self, cpu_group):
        super().__init__(cpu_group)

    def naive_combine_reduce(self, x: torch.Tensor, y: torch.Tensor,
            cu_tokens_across_sp_cpu: torch.Tensor,
            is_sequence_parallel: bool = False):
        assert (len(x.shape) == 2)
        assert (len(y.shape) == 2)
        ep_rank = self.rank if is_sequence_parallel else self.dp_rank
        x_size = x.size(1)
        total_size = x_size + y.size(1)
        buffer = torch.zeros((cu_tokens_across_sp_cpu[-1], total_size),
                            device=x.device, dtype=x.dtype)
        start = 0 if ep_rank == 0 else cu_tokens_across_sp_cpu[ep_rank - 1]
        end = cu_tokens_across_sp_cpu[ep_rank]
        buffer[start:end, :x_size].copy_(x)
        buffer[start:end, x_size:].copy_(y)
        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        buffer = dist_group.all_reduce(buffer)
        buffer_x = buffer[:, :x_size]
        buffer_y = buffer[:, x_size:]
        return buffer_x.contiguous(), buffer_y.contiguous()

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gather hidden_states and router_logits from all dp ranks.
        """
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sp_size = self.tp_group.world_size if is_sequence_parallel else 1
        cu_tokens_across_sp_cpu = dp_metadata.cu_tokens_across_sp(sp_size)

        total_num_tokens = cu_tokens_across_sp_cpu[-1].item()
        if total_num_tokens < 64:
            hidden_states, router_logits = self.naive_combine_reduce(
                hidden_states, router_logits,
                cu_tokens_across_sp_cpu,
                is_sequence_parallel
            )
        else:
            max_tokens_across_dp_cpu = dp_metadata.max_tokens_across_dp_cpu
            sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
            dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
            if not is_sequence_parallel and total_num_tokens > 1024 and \
                max_tokens_across_dp_cpu * self.dp_world_size == total_num_tokens:
                hidden_states = dist_group.all_gather(hidden_states, 0)
                router_logits = dist_group.all_gather(router_logits, 0)
            else:
                assert sizes[dist_group.rank_in_group] == hidden_states.shape[0]
                hidden_states, router_logits = dist_group.all_gatherv(
                    [hidden_states, router_logits],
                    dim=0,
                    sizes=sizes,
                )

        return hidden_states, router_logits

    def combine(self,
                hidden_states: torch.Tensor,
                is_sequence_parallel: bool = False) -> torch.Tensor:
        """
        Reduce hidden_states from all ranks.
        """
        ep_rank = self.rank if is_sequence_parallel else self.dp_rank

        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sp_size = self.tp_group.world_size if is_sequence_parallel else 1
        cu_tokens_across_sp_cpu = dp_metadata.cu_tokens_across_sp(sp_size)

        total_num_tokens = cu_tokens_across_sp_cpu[-1].item()
        if total_num_tokens < 2048:
            sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
            dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
            hidden_states = dist_group.reduce_scatterv(hidden_states, dim=0, sizes=sizes)
            if not is_sequence_parallel:
                hidden_states = get_tp_group().all_reduce(hidden_states)
        else:
            start = 0 if ep_rank == 0 else cu_tokens_across_sp_cpu[ep_rank - 1]
            end = cu_tokens_across_sp_cpu[ep_rank]

            all_hidden_states = get_ep_group().all_reduce(hidden_states)
            hidden_states = all_hidden_states[start:end, :]
        return hidden_states
    

    def destroy(self):
        pass