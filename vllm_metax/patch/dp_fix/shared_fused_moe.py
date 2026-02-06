# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# ------------------------------------------------------------------------
# Note: This file is a patch for dp opt
# ------------------------------------------------------------------------
import torch

from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.shared_fused_moe import SharedFusedMoE


class MacaSharedFusedMoE(SharedFusedMoE):
    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.use_overlapped:
            if self._shared_experts is not None:
                shared_out = self._shared_experts(hidden_states)

                if (
                    get_tensor_model_parallel_world_size() > 1
                    and self.must_reduce_shared_expert_outputs()
                ):
                    shared_out = tensor_model_parallel_all_reduce(shared_out)
            else:
                shared_out = None

            fused_out = FusedMoE.forward(
                self,
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
        else:
            shared_out, fused_out = FusedMoE.forward(
                self,
                hidden_states=hidden_states,
                router_logits=router_logits,
            )
            # ensure early TP reduction of shared expert outputs when required
            if (
                shared_out is not None
                and get_tensor_model_parallel_world_size() > 1
                and self.must_reduce_shared_expert_outputs()
            ):
                shared_out = tensor_model_parallel_all_reduce(shared_out)
        return shared_out, fused_out


SharedFusedMoE.forward = MacaSharedFusedMoE.forward
