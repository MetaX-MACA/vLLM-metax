# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# -----------------------------------------------
# Note: Work around wrong routing weights from mcoplib topk_softmax /
#       topk_sigmoid at the (num_experts=128, topk=8) tile, see issue #270.
#       Expert ids from the kernel are correct in all cases; only the
#       returned weights are wrong at this one shape (softmax returns
#       biased weights when a correction bias is given, sigmoid returns
#       softmax weights). Recompute the weights from gating_output at the
#       kernel-selected ids; the extra gather is gating-only and negligible.
#
# Affected versions: v0.21.0
# -----------------------------------------------
import torch

import vllm._custom_ops as ops

_orig_topk_softmax = ops.topk_softmax
_orig_topk_sigmoid = ops.topk_sigmoid


def _is_bad_tile(gating_output: torch.Tensor, topk_weights: torch.Tensor) -> bool:
    return gating_output.shape[-1] == 128 and topk_weights.shape[-1] == 8


def _renorm_(w: torch.Tensor) -> torch.Tensor:
    return w.div_(w.sum(dim=-1, keepdim=True))


def maca_topk_softmax(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
) -> None:
    _orig_topk_softmax(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize,
        e_score_correction_bias,
    )
    # Without a bias the kernel is correct at every shape.
    if e_score_correction_bias is None or not _is_bad_tile(gating_output, topk_weights):
        return
    # Bias only affects expert selection; weights must be the unbiased softmax.
    scores = torch.softmax(gating_output.float(), dim=-1)
    w = scores.gather(1, topk_ids.long())
    if renormalize:
        _renorm_(w)
    topk_weights.copy_(w)


def maca_topk_sigmoid(
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    token_expert_indices: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool = False,
    e_score_correction_bias: torch.Tensor | None = None,
) -> None:
    _orig_topk_sigmoid(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize,
        e_score_correction_bias,
    )
    if not _is_bad_tile(gating_output, topk_weights):
        return
    scores = torch.sigmoid(gating_output.float())
    w = scores.gather(1, topk_ids.long())
    if renormalize:
        _renorm_(w)
    topk_weights.copy_(w)


# ┌------------------------  Metax Modification -------------------------┐
ops.topk_softmax = maca_topk_softmax
ops.topk_sigmoid = maca_topk_sigmoid
# └------------------------  Metax Modification -------------------------┘
