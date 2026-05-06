# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch

from vllm.distributed import get_dp_group, get_ep_group, get_tp_group
from vllm.forward_context import get_forward_context

from vllm.distributed.device_communicators.base_device_communicator import (
    All2AllManagerBase,
)


class CoArAll2AllManager(All2AllManagerBase):
    """
    A opt implementation of all2all communication.
    It uses all-reduce under the hood, which is not
    efficient at all. The main purpose is for testing and
    debugging.
    """

    def __init__(self, cpu_group):
        super().__init__(cpu_group)

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """
        Gather hidden_states and router_logits from all dp ranks.
        """
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        assert sizes is not None
        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        assert sizes[dist_group.rank_in_group] == hidden_states.shape[0]

        tensors_to_gather = [hidden_states, router_logits]
        if extra_tensors is not None:
            tensors_to_gather.extend(extra_tensors)

        if len(set(sizes)) <= 1 and hidden_states.shape[0] <= 512:
            gathered_tensors = []
            for tensors in tensors_to_gather:
                gathered_tensors.append(dist_group.all_gather(tensors, dim=0))
        else:
            gathered_tensors = dist_group.all_gatherv(
                tensors_to_gather,
                dim=0,
                sizes=sizes,
            )

        if extra_tensors is not None:
            return (gathered_tensors[0], gathered_tensors[1], gathered_tensors[2:])
        return gathered_tensors[0], gathered_tensors[1]

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        """
        Gather hidden_states and router_logits from all dp ranks.
        """
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        assert sizes is not None
        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        assert sizes[dist_group.rank_in_group] == hidden_states.shape[0]

        tensors_to_gather = [hidden_states, topk_weights, topk_ids]
        if extra_tensors is not None:
            tensors_to_gather.extend(extra_tensors)

        if len(set(sizes)) <= 1 and hidden_states.shape[0] <= 512:
            gathered_tensors = []
            for tensors in tensors_to_gather:
                gathered_tensors.append(dist_group.all_gather(tensors, dim=0))
        else:
            gathered_tensors = dist_group.all_gatherv(
                tensors_to_gather,
                dim=0,
                sizes=sizes,
            )

        hidden_states = gathered_tensors[0]
        topk_weights = gathered_tensors[1]
        topk_ids = gathered_tensors[2]

        if extra_tensors is None:
            return hidden_states, topk_weights, topk_ids

        return hidden_states, topk_weights, topk_ids, gathered_tensors[3:]

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
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
            assert sizes is not None
            dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
            hidden_states = dist_group.reduce_scatterv(
                hidden_states, dim=0, sizes=sizes
            )
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
