# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# -----------------------------------------------
# Note: Work around wrong routing weights from mcoplib topk_softmax /
#       topk_sigmoid at the (num_experts=128, topk=8) tile, see issue #270.
#       Expert ids from the kernel are correct in all cases; only the
#       returned weights are wrong at this one shape (softmax returns
#       biased weights when a correction bias is given, sigmoid returns
#       softmax weights on mcoplib <= 0.4.2; sigmoid is fixed in 0.4.5).
#       On the first (128, 8) call each op probes the kernel against a
#       reference once and only recomputes weights when the kernel is
#       wrong, so the patch deactivates itself on fixed kernels.
#
# Affected versions: v0.21.0
# -----------------------------------------------
import torch

import vllm._custom_ops as ops

_orig_topk_softmax = ops.topk_softmax
_orig_topk_sigmoid = ops.topk_sigmoid

_BAD_EXPERTS = 128
_BAD_TOPK = 8
# None = not probed yet, True = kernel wrong at (128, 8), False = kernel ok
_kernel_broken: dict[str, bool | None] = {"softmax": None, "sigmoid": None}


def _is_bad_tile(gating_output: torch.Tensor, topk_weights: torch.Tensor) -> bool:
    return (
        gating_output.shape[-1] == _BAD_EXPERTS
        and topk_weights.shape[-1] == _BAD_TOPK
    )


def _probe(kind: str, device: torch.device) -> bool:
    orig = _orig_topk_softmax if kind == "softmax" else _orig_topk_sigmoid
    g = torch.randn(4, _BAD_EXPERTS, dtype=torch.float32, device=device)
    bias = torch.randn(_BAD_EXPERTS, dtype=torch.float32, device=device)
    w = torch.empty(4, _BAD_TOPK, dtype=torch.float32, device=device)
    idx = torch.empty(4, _BAD_TOPK, dtype=torch.int32, device=device)
    tei = torch.empty(4, _BAD_TOPK, dtype=torch.int32, device=device)
    orig(w, idx, tei, g, False, bias if kind == "softmax" else None)
    act = torch.softmax(g, -1) if kind == "softmax" else torch.sigmoid(g)
    ref = act.gather(1, idx.long())
    return (w - ref).abs().max().item() > 1e-3


def _fixed_weights(
    kind: str,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    gating_output: torch.Tensor,
    renormalize: bool,
) -> None:
    if _kernel_broken[kind] is None:
        _kernel_broken[kind] = _probe(kind, gating_output.device)
    if not _kernel_broken[kind]:
        return
    act = (
        torch.softmax(gating_output.float(), dim=-1)
        if kind == "softmax"
        else torch.sigmoid(gating_output.float())
    )
    w = act.gather(1, topk_ids.long())
    if renormalize:
        w = w.div_(w.sum(dim=-1, keepdim=True))
    topk_weights.copy_(w)


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
    _fixed_weights("softmax", topk_weights, topk_ids, gating_output, renormalize)


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
    _fixed_weights("sigmoid", topk_weights, topk_ids, gating_output, renormalize)


# ┌------------------------  Metax Modification -------------------------┐
ops.topk_softmax = maca_topk_softmax
ops.topk_sigmoid = maca_topk_sigmoid
# └------------------------  Metax Modification -------------------------┘
