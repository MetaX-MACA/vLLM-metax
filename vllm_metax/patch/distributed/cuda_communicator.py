# SPDX-License-Identifier: Apache-2.0

# ------------------------------------------------------------------------
# Note: This file is a patch to opt dp all2all
# ------------------------------------------------------------------------

import torch
from torch.distributed import ProcessGroup

import vllm.envs as envs
from vllm_metax import envs as mx_envs
from vllm.logger import init_logger
from vllm.distributed.device_communicators.pynccl import register_nccl_symmetric_ops
from vllm.distributed.device_communicators.pynccl_allocator import (
    is_symmetric_memory_enabled,
)
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)

logger = init_logger(__name__)


class MacaCommunicator(CudaCommunicator):
    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
    ):
        DeviceCommunicatorBase.__init__(
            self, cpu_group, device, device_group, unique_name
        )
        if "tp" not in unique_name:
            # custom allreduce or torch symm mem can be used only by tp
            use_custom_allreduce = False
            use_torch_symm_mem = False
        else:
            from vllm.distributed.parallel_state import _ENABLE_CUSTOM_ALL_REDUCE

            use_custom_allreduce = _ENABLE_CUSTOM_ALL_REDUCE
            use_torch_symm_mem = envs.VLLM_ALLREDUCE_USE_SYMM_MEM

        self.use_custom_allreduce = use_custom_allreduce
        self.use_torch_symm_mem = use_torch_symm_mem

        # lazy import to avoid documentation build error
        from vllm.distributed.device_communicators.custom_all_reduce import (
            CustomAllreduce,
        )
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
        from vllm.distributed.device_communicators.quick_all_reduce import (
            QuickAllReduce,
        )
        from vllm.distributed.device_communicators.symm_mem import SymmMemCommunicator

        self.pynccl_comm: PyNcclCommunicator | None = None
        if self.world_size > 1:
            self.pynccl_comm = PyNcclCommunicator(
                group=self.cpu_group,
                device=self.device,
            )
            if is_symmetric_memory_enabled():
                register_nccl_symmetric_ops(self.pynccl_comm)

        self.ca_comm: CustomAllreduce | None = None
        self.qr_comm: QuickAllReduce | None = None
        self.symm_mem_comm: SymmMemCommunicator | None = None

        if self.use_all2all:
            if self.all2all_backend == "naive":
                from vllm.distributed.device_communicators.all2all import (
                    NaiveAll2AllManager,
                )

                self.all2all_manager = NaiveAll2AllManager(self.cpu_group)
            elif self.all2all_backend == "allgather_reducescatter":
                from vllm.distributed.device_communicators.all2all import (
                    AgRsAll2AllManager,
                )

                self.all2all_manager = AgRsAll2AllManager(self.cpu_group)
            elif self.all2all_backend == "pplx":
                from vllm.distributed.device_communicators.all2all import (
                    PPLXAll2AllManager,
                )

                self.all2all_manager = PPLXAll2AllManager(self.cpu_group)
            elif self.all2all_backend == "deepep_high_throughput":
                from vllm.distributed.device_communicators.all2all import (
                    DeepEPHTAll2AllManager,
                )

                self.all2all_manager = DeepEPHTAll2AllManager(self.cpu_group)
            elif self.all2all_backend == "deepep_low_latency":
                from vllm.distributed.device_communicators.all2all import (
                    DeepEPLLAll2AllManager,
                )

                self.all2all_manager = DeepEPLLAll2AllManager(self.cpu_group)
            elif self.all2all_backend == "flashinfer_all2allv":
                from vllm.distributed.device_communicators.all2all import (
                    FlashInferAllToAllManager,
                )

                self.all2all_manager = FlashInferAllToAllManager(self.cpu_group)
            else:
                raise ValueError(f"Unknown all2all backend: {self.all2all_backend}")

            # /------------------------  Metax Modification -------------------------\
            if mx_envs.MACA_DP_OPT and (
                self.all2all_backend == "naive"
                or self.all2all_backend == "allgather_reducescatter"
            ):
                from vllm_metax.distributed.device_communicators.all2all import (
                    CoArAll2AllManager,
                )

                self.all2all_manager = CoArAll2AllManager(self.cpu_group)
                logger.info("Using combine_allreduce manager for opt.")
            # \------------------------  Metax Modification -------------------------/

            logger.info_once(
                "Using %s all2all manager.",
                self.all2all_manager.__class__.__name__,
                scope="global",
            )


CudaCommunicator.__init__ = MacaCommunicator.__init__
