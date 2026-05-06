# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch

from vllm_metax import _custom_ops as mx_ops
from vllm_metax import envs as mx_envs
from vllm import envs as envs
from vllm.model_executor.layers.batch_invariant import (
    vllm_is_batch_invariant,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
    fused_topk_bias,
)
from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk
from vllm.model_executor.utils import maybe_disable_graph_partition
from vllm.platforms import current_platform
from vllm.model_executor.layers.fused_moe.router.grouped_topk_router import (
    GroupedTopKRouter,
)


def fused_grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    e_score_correction_bias: torch.Tensor,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert hidden_states.size(0) == gating_output.size(0), "Number of tokens mismatch"

    if scoring_func == "sigmoid":
        # Fully fused kernel path for sigmoid
        # ┌------------------------  Metax Modification -------------------------┐
        deepseek = (topk == 8 and num_expert_group == 8 and topk_group == 4) or (
            topk == 9 and num_expert_group == 8 and topk_group == 4
        )
        kimi = (topk == 8 and num_expert_group == 1 and topk_group == 1) or (
            topk == 9 and num_expert_group == 1 and topk_group == 1
        )
        if (
            mx_envs.VLLM_METAX_USE_SGL_FUSED_MOE_GROUPED_TOPK
            and (gating_output.shape[1] // num_expert_group <= 384)
            and (kimi or deepseek)
        ):
            bias_bf16 = e_score_correction_bias.to(torch.bfloat16)
            topk_values = torch.empty(
                (hidden_states.shape[0], topk),
                dtype=torch.float,
                device=hidden_states.device,
            )
            topk_indices = torch.empty(
                (hidden_states.shape[0], topk),
                dtype=torch.int,
                device=hidden_states.device,
            )
            mx_ops.sgl_fused_moe_gate_opt(
                gating_output,
                bias_bf16,
                topk_values,
                topk_indices,
                topk,
                renormalize,
                num_expert_group,
                topk_group,
                0,
                routed_scaling_factor,
            )
        else:
            topk_values, topk_indices = mx_ops.grouped_topk(
                gating_output,  # raw logits
                num_expert_group,
                topk_group,
                topk,
                renormalize,
                routed_scaling_factor,
                e_score_correction_bias,
                1,  # scoring_func=1 for sigmoid
            )
    # └------------------------- Metax Modification -------------------------┘
    elif scoring_func == "softmax":
        # Apply softmax in Python, then use fused kernel
        # TODO: Add support for softmax in kernel
        scores = torch.softmax(gating_output, dim=-1)
        # ┌------------------------  Metax Modification -------------------------┐
        topk_values, topk_indices = mx_ops.grouped_topk(
            # └------------------------- Metax Modification -------------------------┘
            scores,  # pre-computed scores
            num_expert_group,
            topk_group,
            topk,
            renormalize,
            routed_scaling_factor,
            e_score_correction_bias,
            0,  # scoring_func=0 (no activation, scores already computed)
        )
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    # Fused kernel outputs float32 values and int32 indices directly
    return topk_values, topk_indices


# This is used by the Deepseek-V2 and Deepseek-V3 model
@torch.compile(
    dynamic=True,
    backend=current_platform.simple_compile_backend,
    options=maybe_disable_graph_partition(current_platform.simple_compile_backend),
)
def grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        envs.VLLM_USE_FUSED_MOE_GROUPED_TOPK
        and num_expert_group <= 32
        and topk <= 32
        and e_score_correction_bias is not None
    ):
        return fused_grouped_topk(
            hidden_states=hidden_states,
            gating_output=gating_output,
            topk=topk,
            renormalize=renormalize,
            e_score_correction_bias=e_score_correction_bias,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            scoring_func=scoring_func,
            routed_scaling_factor=routed_scaling_factor,
        )

    assert hidden_states.size(0) == gating_output.size(0), "Number of tokens mismatch"
    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    num_token = scores.size(0)
    if e_score_correction_bias is not None:
        # Store original scores before applying correction bias. We use biased
        # scores for expert selection but original scores for routing weights
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores.view(num_token, num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        )
    else:
        group_scores = (
            scores.view(num_token, num_expert_group, -1).max(dim=-1).values
        )  # [n, n_group]

    # For batch invariance, use sorted=True to ensure deterministic expert selection
    use_sorted = vllm_is_batch_invariant()
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=use_sorted)[
        1
    ]  # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.size(-1) // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))  # [n, e]

    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=use_sorted)[1]
        # Use original unbiased scores for the routing weights
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(
            tmp_scores, k=topk, dim=-1, sorted=use_sorted
        )

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


class MacaGroupedTopKRouter(GroupedTopKRouter):
    """Router using grouped top-k routing (e.g., DeepSeekV2/V3)."""

    def _compute_routing(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        indices_type: torch.dtype | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute routing using grouped top-k."""

        def valid_grouping() -> bool:
            # Check if num_experts is greater than num_expert_group
            # and is divisible by num_expert_group
            num_experts = router_logits.shape[-1]
            if num_experts <= self.num_expert_group:
                return False
            return num_experts % self.num_expert_group == 0

        if not valid_grouping():
            if self.e_score_correction_bias is not None:
                topk_weights, topk_ids = fused_topk_bias(
                    hidden_states=hidden_states,
                    gating_output=router_logits,
                    e_score_correction_bias=self.e_score_correction_bias.data,
                    topk=self.top_k,
                    renormalize=self.renormalize,
                )
                if self.routed_scaling_factor != 1.0:
                    topk_weights *= self.routed_scaling_factor
            else:
                topk_weights, topk_ids, token_expert_indices = fused_topk(
                    hidden_states=hidden_states,
                    gating_output=router_logits,
                    topk=self.top_k,
                    renormalize=self.renormalize,
                    indices_type=indices_type,
                )
            return topk_weights, topk_ids

        topk_weights, topk_ids = grouped_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            num_expert_group=self.num_expert_group,
            topk_group=self.topk_group,
            scoring_func=self.scoring_func,
            routed_scaling_factor=self.routed_scaling_factor,
            e_score_correction_bias=self.e_score_correction_bias,
        )

        return topk_weights, topk_ids


GroupedTopKRouter._compute_routing = MacaGroupedTopKRouter._compute_routing
