# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch

import vllm.envs as envs
import vllm_metax.envs as mx_envs

from vllm.platforms import current_platform

from vllm.distributed.device_communicators.all2all import (
    AgRsAll2AllManager,
    DeepEPLLAll2AllManager,
    DeepEPHTAll2AllManager,
)


class MacaAgRsAll2AllManager(AgRsAll2AllManager):
    """
    An implementation of all2all communication based on
    all-gather (dispatch) and reduce-scatter (combine).
    """

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
        """Gather hidden_states and router_logits from all dp ranks."""
        dist_group = self._get_comm_group(is_sequence_parallel)
        sizes = self._get_sizes(hidden_states.shape[0], dist_group)
        assert sizes[dist_group.rank_in_group] == hidden_states.shape[0]

        tensors_to_gather = [hidden_states, router_logits]
        if extra_tensors is not None:
            tensors_to_gather.extend(extra_tensors)

        gathered_tensors = dist_group.all_gatherv(
            tensors_to_gather,
            dim=0,
            sizes=sizes,
        )

        if extra_tensors is not None:
            return (gathered_tensors[0], gathered_tensors[1], gathered_tensors[2:])
        return gathered_tensors[0], gathered_tensors[1]

    @torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
    def _pack(
        self,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        topk_id_float = topk_ids.to(torch.float32)

        if extra_tensors is not None and len(extra_tensors) == 1:
            combined = torch.cat([topk_weights, topk_id_float, extra_tensors[0]], dim=1)
        else:
            combined = torch.cat([topk_weights, topk_id_float], dim=1)
        return combined

    def _unpack(
        self,
        topk: int,
        gathered_tensors: list[torch.Tensor],
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        | tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]
    ):
        hidden_states = gathered_tensors[0]
        packed = gathered_tensors[1]
        n = packed.shape[1] - 2 * topk
        m = packed.shape[0]
        topk_weights = torch.empty(
            m, topk, device=hidden_states.device, dtype=torch.float32
        )
        topk_ids = torch.empty(m, topk, device=hidden_states.device, dtype=torch.int32)
        scale = torch.empty(m, n, device=hidden_states.device, dtype=torch.float32)
        torch.ops._C.fused_unpack(packed, topk, n, topk_weights, topk_ids, scale)

        if extra_tensors is None:
            return hidden_states, topk_weights, topk_ids

        if len(extra_tensors) == 1:
            return hidden_states, topk_weights, topk_ids, [scale]

        return hidden_states, topk_weights, topk_ids, gathered_tensors[2:]

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

        combined:
        ┌─────────────┬──────────┬─────────┐
        │ weights: K  │ ids: K   │ scale:S │
        └─────────────┴──────────┴─────────┘

        shape = [M_local, 2K + S]
        """
        if mx_envs.VLLM_METAX_EXPERIMENTAL_MOE_BYTE_PACK:
            from .byte_pack import gather_packed, should_pack

            tensors = [hidden_states, topk_weights, topk_ids]
            if extra_tensors is not None:
                tensors.extend(extra_tensors)
            # As with all_gatherv, non-token shape/dtype and the environment
            # setting must agree across ranks. Strides and local N may differ.
            dist_group = self._get_comm_group(is_sequence_parallel)
            sizes = self._get_sizes(hidden_states.shape[0], dist_group)
            assert sizes[dist_group.rank_in_group] == hidden_states.shape[0]
            # Uniform batches and two-rank eager execution regressed on HPCC.
            # The opt-in is experimental, not a MACA performance guarantee.
            if should_pack(tensors, sizes):
                outputs = gather_packed(tensors, sizes, dist_group.all_gatherv)
                if extra_tensors is not None:
                    return outputs[0], outputs[1], outputs[2], outputs[3:]
                return outputs[0], outputs[1], outputs[2]

        # NOTE(MetaX): fused_unpack writes FP32 weights/scales and INT32 IDs.
        # Upstream also dispatches packed integer scales and arbitrary extra
        # tensors; preserve their dtype/shape through the generic collective.
        # Keep packing only for the layout supported by the installed kernel.
        if (
            # NOTE(metax): fused_unpack is MCOPLIB-only. The source build
            # retains arbitrary tensor dtypes through the native collective.
            not mx_envs.USE_PRECOMPILED_KERNEL
            or topk_weights.dtype != torch.float32
            or topk_ids.dtype != torch.int32
            or (
                extra_tensors is not None
                and len(extra_tensors) == 1
                and (
                    extra_tensors[0].ndim != 2
                    or extra_tensors[0].dtype != torch.float32
                )
            )
        ):
            return super().dispatch(
                hidden_states,
                topk_weights,
                topk_ids,
                is_sequence_parallel,
                extra_tensors,
            )

        dist_group = self._get_comm_group(is_sequence_parallel)
        sizes = self._get_sizes(hidden_states.shape[0], dist_group)
        assert sizes[dist_group.rank_in_group] == hidden_states.shape[0]

        topk = topk_weights.shape[1]
        combined = self._pack(topk_weights, topk_ids, extra_tensors)
        tensors_to_gather = [hidden_states, combined]

        if extra_tensors is not None and len(extra_tensors) != 1:
            tensors_to_gather.extend(extra_tensors)

        gathered_tensors = dist_group.all_gatherv(
            tensors_to_gather,
            dim=0,
            sizes=sizes,
        )

        return self._unpack(topk, gathered_tensors, extra_tensors)

    def destroy(self):
        pass


class MacaDeepEPHTAll2AllManager(DeepEPHTAll2AllManager):
    def _make_all2all_kwargs(self) -> dict[Any, Any]:
        import os

        assert os.getenv("MXSHMEM_LIB_PATH", None) is not None, (
            "please setting MXSHMEM_LIB_PATH and add ${MXSHMEM_LIB_PATH} into LD_LIBRARY_PATH"
        )

        # Defaults for internode and intranode are taken from DeepEP tests.
        num_nvl_bytes = envs.VLLM_DEEPEP_BUFFER_SIZE_MB * 1024 * 1024
        num_rdma_bytes = None
        num_qps_per_rank = None

        if self.internode and not envs.VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE:
            num_rdma_bytes = envs.VLLM_DEEPEP_BUFFER_SIZE_MB * 1024 * 1024
            num_qps_per_rank = self.num_sms // 2
        else:
            num_rdma_bytes = 0
            num_qps_per_rank = 1

        assert num_rdma_bytes is not None
        assert num_qps_per_rank is not None
        # TODO: remove platform-specific logic
        # once ROCm DeepEP is updated with the latest APIs.
        kwargs = dict(
            group=self.cpu_group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=False,
            num_qps_per_rank=num_qps_per_rank,
            # /------------------- Metax Modification -----------------------\
            # explicitly_destroy=True,
            # /------------------- Metax Modification -----------------------\
        )
        return kwargs

    def destroy(self):
        with self.handle_cache._lock:
            # /------------------- Metax Modification -----------------------\
            # /---Do not call Buffer.destroy because explicitly_destroy=True is not supported---\

            # for _, handle in self.handle_cache._cache.items():
            #     handle.destroy()
            # \------------------- Metax Modification -----------------------/
            self.handle_cache._cache.clear()


class MacaDeepEPLLAll2AllManager(DeepEPLLAll2AllManager):
    def _make_all2all_kwargs(
        self,
        max_num_tokens_per_dp_rank: int,
        token_hidden_size: int,
        num_ep_ranks: int,
        num_global_experts: int,
        num_local_experts: int,
    ) -> dict[Any, Any]:
        """max_num_tokens_per_dp_rank : the maximum number of tokens a DP rank
          can dispatch all the ranks must hold the same value.
        token_hidden_size: the hidden dimension of each token.
        num_ep_ranks: the number of EP group ranks.
        num_global_experts: Number of experts in the model.
        num_local_experts: Number of experts in an EP rank.
        """
        import os

        assert os.getenv("MXSHMEM_LIB_PATH", None) is not None, (
            "please setting MXSHMEM_LIB_PATH and add ${MXSHMEM_LIB_PATH} into LD_LIBRARY_PATH"
        )

        import deep_ep  # type: ignore[import-not-found]

        # Defaults for internode and intranode are taken from DeepEP tests.
        num_nvl_bytes = envs.VLLM_DEEPEP_BUFFER_SIZE_MB * 1024 * 1024
        num_qps_per_rank = num_local_experts
        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
            num_max_dispatch_tokens_per_rank=max_num_tokens_per_dp_rank,
            hidden=token_hidden_size,
            num_ranks=num_ep_ranks,
            num_experts=num_global_experts,
        )

        assert num_rdma_bytes is not None

        return dict(
            group=self.cpu_group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=num_qps_per_rank,
            # /------------------- Metax Modification -----------------------\
            # allow_nvlink_for_low_latency_mode=True,
            # allow_mnnvl=envs.VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL,
            # \------------------- Metax Modification -----------------------/
        )

    def destroy(self):
        with self.handle_cache._lock:
            # /------------------- Metax Modification -----------------------\
            # /---Do not call Buffer.destroy because explicitly_destroy=True is not supported---\

            # for _, handle in self.handle_cache._cache.items():
            #     handle.destroy()
            # \------------------- Metax Modification -----------------------/
            self.handle_cache._cache.clear()
