# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from vllm.model_executor.layers.mamba.linear_attn import (
    MiniMaxText01RMSNormTP as vllm_MiniMaxText01RMSNormTP,
)

from functools import partial
import torch
from torch import nn
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)


@vllm_MiniMaxText01RMSNormTP.register_oot
class MiniMaxText01RMSNormTP(vllm_MiniMaxText01RMSNormTP):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        weight_shard_world_size: int | None = None,
        weight_shard_rank: int | None = None,
    ) -> None:
        super(vllm_MiniMaxText01RMSNormTP, self).__init__()
        self.tp_world = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        self.weight_shard_world = weight_shard_world_size or self.tp_world
        self.weight_shard_rank = (
            self.tp_rank if weight_shard_rank is None else weight_shard_rank
        )

        self.weight = nn.Parameter(torch.ones(hidden_size // self.weight_shard_world))
        self.weight.weight_loader = partial(
            self.weight_loader,
            shard_world_size=self.weight_shard_world,
            shard_rank=self.weight_shard_rank,
        )
        self.variance_epsilon = eps

    @staticmethod
    def weight_loader(
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        shard_world_size: int | None = None,
        shard_rank: int | None = None,
    ) -> None:
        if shard_world_size is None:
            shard_world_size = get_tensor_model_parallel_world_size()
        if shard_rank is None:
            shard_rank = get_tensor_model_parallel_rank()

        shard_size = loaded_weight.shape[0] // shard_world_size
        shard = slice(shard_rank * shard_size, (shard_rank + 1) * shard_size)
        param.data.copy_(loaded_weight[shard])
