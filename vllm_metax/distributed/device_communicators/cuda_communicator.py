# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# ------------------------------------------------------------------------
# Note: This file is a patch to opt dp all2all
# ------------------------------------------------------------------------

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
    ):
        super().__init__(
            cpu_group,
            device,
            device_group,
            unique_name,
            global_ranks,
            global_world_size,
            tcp_store_group,
        )
        # /------------------------  Metax Modification -------------------------\
        if self.use_all2all:
            if (
                mx_envs.VLLM_METAX_OPTIMIZED_DP_ALL2ALL
                and self.all2all_backend == "allgather_reducescatter"
            ):
                from .all2all import MacaAgRsAll2AllManager

                self.all2all_manager = MacaAgRsAll2AllManager(self.cpu_group)
                logger.info_once(
                    "Maca override AgRsAll2AllManager to %s for better performance.",
                    self.all2all_manager.__class__.__name__,
                )
            elif self.all2all_backend == "deepep_low_latency":
                from .all2all import MacaDeepEPLLAll2AllManager

                self.all2all_manager = MacaDeepEPLLAll2AllManager(
                    self.cpu_group, tcp_store_group
                )
                logger.info_once(
                    "Maca override DeepEPLLAll2AllManager to %s for better performance.",
                    self.all2all_manager.__class__.__name__,
                )
        # \------------------------  Metax Modification -------------------------/
