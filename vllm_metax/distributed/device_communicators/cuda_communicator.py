# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# ------------------------------------------------------------------------
# Note: This file is a patch to opt dp all2all
# ------------------------------------------------------------------------

import torch
from torch.distributed import ProcessGroup

from vllm_metax import envs as mx_envs
from vllm.logger import init_logger
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.distributed.utils import StatelessProcessGroup

logger = init_logger(__name__)


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
        if (
            self.use_all2all
            and mx_envs.VLLM_METAX_DP_OPT
            and (
                self.all2all_backend == "naive"
                or self.all2all_backend == "allgather_reducescatter"
            )
        ):
            from vllm_metax.distributed.device_communicators.all2all import (
                CoArAll2AllManager,
            )

            self.all2all_manager = CoArAll2AllManager(self.cpu_group)
            logger.info_once(
                "Maca switch all2all_backend to %s all2all manager for better performance.",
                self.all2all_manager.__class__.__name__,
                scope="global",
            )
        # \------------------------  Metax Modification -------------------------/
