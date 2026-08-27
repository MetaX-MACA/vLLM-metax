# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# -----------------------------------------------
# Note: Alias the Modular MoE fused output to the final output on MetaX. This
#       removes the redundant copy in TopKWeightAndReduceNoOP for every
#       compatible quantization method.
#
# Affected versions: v0.24.0
#
# Remove at: Remove after the supported public vLLM version provides this alias
#            path for MetaX Modular MoE experts.
# -----------------------------------------------
"""Remove redundant output copies from MetaX Modular MoE kernels."""

import torch

from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    ExpertTokensMetadata,
    FusedMoEKernelModularImpl,
)


class MacaFusedMoEKernelModularImpl(FusedMoEKernelModularImpl):
    def _fused_experts(
        self,
        in_dtype: torch.dtype,
        a1q: torch.Tensor,
        a1q_scale: torch.Tensor | None,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        local_num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        expert_tokens_meta: ExpertTokensMetadata | None,
        output_alias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _, M_full, N, K, top_k = self.fused_experts.moe_problem_size(
            a1q, w1, w2, topk_ids
        )

        # No tokens can be returned by an all-to-all dispatch on this rank.
        if M_full == 0:
            return torch.empty_like(a1q, dtype=in_dtype)

        workspace13, workspace2, fused_out = self._allocate_buffers(
            in_dtype,
            a1q.device,
            M_full,
            M_full,
            N,
            K,
            top_k,
            global_num_experts,
            local_num_experts,
            expert_tokens_meta,
            activation,
        )

        # /------------------------ MetaX Modification -------------------------\
        # Reuse the final output when it is compatible with fused_out to avoid memcpy.
        use_output_alias = (
            output_alias is not None
            and output_alias.shape == fused_out.shape
            and output_alias.dtype == fused_out.dtype
            and output_alias.device == fused_out.device
            and output_alias.is_contiguous()
        )

        if use_output_alias:
            assert output_alias is not None
            fused_out = output_alias
        # \------------------------ MetaX Modification -------------------------/

        self.fused_experts.apply(
            output=fused_out,
            hidden_states=a1q,
            w1=w1,
            w2=w2,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=activation,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            a1q_scale=a1q_scale,
            a2_scale=self.fused_experts.a2_scale,
            workspace13=workspace13,
            workspace2=workspace2,
            expert_tokens_meta=expert_tokens_meta,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )

        return fused_out


FusedMoEKernelModularImpl._fused_experts = MacaFusedMoEKernelModularImpl._fused_experts
