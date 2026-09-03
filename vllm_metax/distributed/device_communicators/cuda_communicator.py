# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# ------------------------------------------------------------------------
# Note: This file is for optimized dp ep all2all
#
# TODO(2026.Sept.1st): mccl all_gather hang
# ------------------------------------------------------------------------


from vllm.distributed.device_communicators.all_reduce_utils import (
    should_nccl_symm_mem_ag_rs,
)
from vllm.platforms import current_platform

import torch
from torch.distributed import ProcessGroup

from vllm_metax import envs as mx_envs
from vllm.distributed.device_communicators.cuda_communicator import (
    CudaCommunicator,
    logger,
)
from vllm.distributed.utils import StatelessProcessGroup


class MacaCommunicator(CudaCommunicator):
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
        global_ranks: list[int] | None = None,
        global_world_size: int | None = None,
        tcp_store_group: StatelessProcessGroup | None = None,
        use_all2all: bool = False,
    ):
        super().__init__(
            cpu_group,
            device,
            device_group,
            unique_name,
            global_ranks,
            global_world_size,
            tcp_store_group,
            use_all2all=use_all2all,
        )
        # /------------------------  Metax Modification -------------------------\
        if self.use_all2all:
            if mx_envs.VLLM_METAX_OPTIMIZED_DP_ALL2ALL and self.all2all_backend in (
                "naive",
                "allgather_reducescatter",
            ):
                from .all2all import MacaAgRsAll2AllManager

                self.all2all_manager = MacaAgRsAll2AllManager(
                    self.cpu_group, tcp_store_group
                )
                logger.info_once(
                    "Maca override AgRsAll2AllManager to %s for better performance.",
                    self.all2all_manager.__class__.__name__,
                )
            elif self.all2all_backend == "deepep_high_throughput":
                from .all2all import MacaDeepEPHTAll2AllManager

                self.all2all_manager = MacaDeepEPHTAll2AllManager(
                    self.cpu_group, tcp_store_group
                )
                logger.info_once(
                    "Maca override DeepEPHTAll2AllManager to %s for better performance or compatibility.",
                    self.all2all_manager.__class__.__name__,
                )
            elif self.all2all_backend == "deepep_low_latency":
                from .all2all import MacaDeepEPLLAll2AllManager

                self.all2all_manager = MacaDeepEPLLAll2AllManager(
                    self.cpu_group, tcp_store_group
                )
                logger.info_once(
                    "Maca override DeepEPLLAll2AllManager to %s for better performance or compatibility.",
                    self.all2all_manager.__class__.__name__,
                )
        # \------------------------  Metax Modification -------------------------/

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # Route uniform dim-0 all-gathers through NVLS symmetric memory when
        # enabled (mirrors reduce_scatter); otherwise fall back to the
        # PyNccl/base-class all-gather. Sequence parallelism's
        # gather-before-GEMM uses dim=0 with tp-aligned (uniform) shards.
        if dim < 0:
            dim += input_.dim()
        if dim == 0 and should_nccl_symm_mem_ag_rs():
            return self._all_gather_symm_mem(input_.contiguous())

        pynccl_comm = self.pynccl_comm
        if pynccl_comm is None or pynccl_comm.disabled:
            return super(CudaCommunicator, self).all_gather(input_, dim)

        # On ROCm, the base-class all_gather (all_gather_into_tensor) is faster
        # than the manual pynccl + torch.empty + movedim + reshape path below,
        # which adds a per-call output allocation and (for dim != 0) an extra
        # copy on every step. This is on the hot path for TP forward passes, so
        # keep ROCm on the base-class collective to avoid a decode regression.
        # /------------------------  Metax Modification -------------------------\
        # workaround for the issue of multi node inference hanging
        if current_platform.is_rocm() or current_platform.is_cuda_alike():
            return super(CudaCommunicator, self).all_gather(input_, dim)
        # \------------------------  Metax Modification -------------------------/

        input_size = input_.size()
        output_size = (input_size[0] * self.world_size,) + input_size[1:]
        output_tensor = torch.empty(
            output_size, dtype=input_.dtype, device=input_.device
        )
        pynccl_comm.all_gather(output_tensor, input_.contiguous())
        output_tensor = output_tensor.reshape((self.world_size,) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        return output_tensor.reshape(
            input_size[:dim]
            + (self.world_size * input_size[dim],)
            + input_size[dim + 1 :]
        )
