# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# ------------------------------------------------------------
# Note: 替换 MiniMaxText01RMSNormTP.__init__，将其中的
#       get_allreduce_workspace 导入改为 metax 定制版本。
# ------------------------------------------------------------
from functools import partial
import torch
from torch import nn
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)
from vllm.model_executor.layers.minimax_rms_norm.rms_norm_tp import (
    MINIMAX_QK_NORM_MAX_TOKEN_NUM,
    MiniMaxText01RMSNormTP,
    _MINIMAX_FUSED_AR_RMS_QK,
)

# ┌------------------------  Metax Modification -------------------------┐
from vllm_metax.patch.model_executor.layers.lamport_workspace import (
    get_allreduce_workspace as _metax_get_allreduce_workspace,
)


# └------------------------- Metax Modification -------------------------┘
def _new_init(
    self,
    hidden_size: int,
    eps: float = 1e-6,
    *,
    weight_shard_world_size: int | None = None,
    weight_shard_rank: int | None = None,
) -> None:
    super(MiniMaxText01RMSNormTP, self).__init__()
    self.tp_world = get_tensor_model_parallel_world_size()
    self.tp_rank = get_tensor_model_parallel_rank()
    self.weight_shard_world = weight_shard_world_size or self.tp_world
    self.weight_shard_rank = (
        self.tp_rank if weight_shard_rank is None else weight_shard_rank
    )
    self.weight = nn.Parameter(torch.ones(hidden_size // self.weight_shard_world))
    self.weight.weight_loader = partial(
        MiniMaxText01RMSNormTP.weight_loader,
        shard_world_size=self.weight_shard_world,
        shard_rank=self.weight_shard_rank,
    )
    self.variance_epsilon = eps
    self.workspace = None
    if _MINIMAX_FUSED_AR_RMS_QK is not None and self.tp_world > 1:
        # ┌------------------------  Metax Modification -------------------------┐
        self.workspace = _metax_get_allreduce_workspace(
            rank=self.tp_rank,
            world_size=self.tp_world,
            max_tokens=MINIMAX_QK_NORM_MAX_TOKEN_NUM,
            process_group=get_tp_group().cpu_group,
        )
        # └------------------------- Metax Modification -------------------------┘


MiniMaxText01RMSNormTP.__init__ = _new_init
