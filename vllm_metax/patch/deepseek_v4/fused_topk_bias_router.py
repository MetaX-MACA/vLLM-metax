# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
from collections.abc import Callable

import torch
import torch.nn.functional as F

import vllm._custom_ops as ops
import vllm.envs as envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.distributed.eplb.eplb_state import EplbLayerState
from vllm.model_executor.layers.fused_moe.config import (
    RoutingMethodType,
    get_routing_method_type,
)

from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    vllm_topk_softmax,
    vllm_topk_sigmoid,
    FusedTopKBiasRouter,
)


def fused_topk_bias(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    scoring_func: str,
    e_score_correction_bias: torch.Tensor,
    topk: int,
    renormalize: bool,
    indices_type: torch.dtype | None = None,
    input_tokens: torch.Tensor | None = None,
    hash_indices_table: torch.Tensor | None = None,
    routed_scaling_factor: float = 1.0,
):
    if not rocm_aiter_ops.is_fused_moe_enabled():
        assert hidden_states.size(0) == gating_output.size(0), (
            "Number of tokens mismatch"
        )

        M, _ = hidden_states.size()

        topk_weights = torch.empty(
            M, topk, dtype=torch.float32, device=hidden_states.device
        )
        topk_ids = torch.empty(
            M,
            topk,
            dtype=torch.int32 if indices_type is None else indices_type,
            device=hidden_states.device,
        )
        token_expert_indices = torch.empty(
            M, topk, dtype=torch.int32, device=hidden_states.device
        )

        if scoring_func == "softmax":
            topk_weights, topk_ids = vllm_topk_softmax(
                topk_weights,
                topk_ids,
                token_expert_indices,
                gating_output,
                renormalize,
                e_score_correction_bias,
            )
            if routed_scaling_factor != 1.0:
                topk_weights *= routed_scaling_factor
            return topk_weights, topk_ids
        elif scoring_func == "sigmoid":
            topk_weights, topk_ids = vllm_topk_sigmoid(
                topk_weights,
                topk_ids,
                token_expert_indices,
                gating_output,
                renormalize,
                e_score_correction_bias,
            )
            if routed_scaling_factor != 1.0:
                topk_weights *= routed_scaling_factor
            return topk_weights, topk_ids
        elif scoring_func == "sqrtsoftplus":
            assert False, "not support sqrtsoftplus"
        else:
            raise ValueError(f"Unsupported scoring function: {scoring_func}")

    n_routed_experts = gating_output.shape[-1]
    if scoring_func == "softmax":
        scores = gating_output.softmax(dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    elif scoring_func == "sqrtsoftplus":
        scores = F.softplus(gating_output).sqrt()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")
    if e_score_correction_bias is not None:
        scores_for_choice = scores.view(
            -1, n_routed_experts
        ) + e_score_correction_bias.unsqueeze(0)
    else:
        scores_for_choice = scores.view(-1, n_routed_experts)
    # For batch invariance, use sorted=True to ensure deterministic expert selection
    if hash_indices_table is not None:
        topk_indices = hash_indices_table[input_tokens]
    else:
        use_sorted = envs.VLLM_BATCH_INVARIANT
        topk_indices = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=use_sorted)[
            1
        ]
    topk_weights = scores.gather(1, topk_indices)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(torch.float32)
    if routed_scaling_factor != 1.0:
        topk_weights *= routed_scaling_factor
    return topk_weights, topk_indices.to(
        torch.int32 if indices_type is None else indices_type
    )


def _torch_topk_softplus_sqrt(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    e_score_correction_bias: torch.Tensor | None = None,
    input_ids: torch.Tensor | None = None,
    hash_indices_table: torch.Tensor | None = None,
):
    scores = F.softplus(gating_output.float()).sqrt()
    original_scores = scores
    if e_score_correction_bias is not None:
        scores_for_choice = scores + e_score_correction_bias.unsqueeze(0)
    else:
        scores_for_choice = scores

    if hash_indices_table is not None:
        assert input_ids is not None
        topk_ids = hash_indices_table[input_ids.long()]
    else:
        topk_ids = torch.topk(scores_for_choice, k=topk, dim=-1, sorted=True)[1]

    topk_weights = original_scores.gather(1, topk_ids.long())
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


class MacaFusedTopKBiasRouter(FusedTopKBiasRouter):
    """Router using fused top-k with e_score_correction_bias."""

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
        *,
        input_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute routing using fused top-k with bias."""
        if self.scoring_func == "sqrtsoftplus":
            topk_weights, topk_ids = _torch_topk_softplus_sqrt(
                gating_output=router_logits,
                e_score_correction_bias=self.e_score_correction_bias.data
                if self.e_score_correction_bias is not None
                else None,
                topk=self.top_k,
                renormalize=self.renormalize,
                input_ids=input_ids,
                hash_indices_table=self._hash_indices_table,
                routed_scaling_factor=self.routed_scaling_factor,
            )
        else:
            topk_weights, topk_ids = fused_topk_bias(
                hidden_states=hidden_states,
                gating_output=router_logits,
                scoring_func=self.scoring_func,
                e_score_correction_bias=self.e_score_correction_bias.data
                if self.e_score_correction_bias is not None
                else None,
                topk=self.top_k,
                renormalize=self.renormalize,
                indices_type=indices_type,
                input_tokens=input_ids,
                hash_indices_table=self._hash_indices_table,
                routed_scaling_factor=self.routed_scaling_factor,
            )

        return topk_weights, topk_ids


FusedTopKBiasRouter._compute_routing = MacaFusedTopKBiasRouter._compute_routing
