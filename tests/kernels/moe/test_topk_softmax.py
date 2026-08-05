# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical correctness tests for the MetaX MoE top-k gating kernels.

Covers the two ``_moe_C`` gating ops used by the generic (non-grouped) fused-MoE
routing path that vLLM-metax inherits from vLLM (``fused_moe.fused_topk`` ->
``ops.topk_softmax``):

* ``_moe_C.topk_softmax`` -- softmax-scored gating
* ``_moe_C.topk_sigmoid`` -- sigmoid-scored gating

Both run the real kernel on MetaX hardware and compare against a PyTorch
reference whose bias semantics mirror vLLM-metax's own ``maca_grouped_topk``
(biased scores select the experts; unbiased scores become the routing weights).

The module skips cleanly when the compiled ``_moe_C`` extension (and therefore a
MetaX backend) is unavailable, so it is safe to collect on non-MetaX CI.
"""

import pytest
import torch

# The MoE gating ops live in the compiled ``_moe_C`` extension, loaded by
# ``MacaPlatform.import_kernels`` via ``import mcoplib._moe_C``. Importing it here
# registers ``torch.ops._moe_C.*`` and skips the whole module cleanly elsewhere.
# (We import the extension directly rather than ``vllm_metax._custom_ops`` so the
# guard is a clean ImportError off-MetaX instead of a registration-time error.)
pytest.importorskip(
    "mcoplib._moe_C",
    reason="mcoplib._moe_C extension not available (non-MetaX backend)",
)


def _moe_op(name: str):
    """Return the named _moe_C op, or None if it (or CUDA) is unavailable."""
    if not torch.cuda.is_available():
        return None
    try:
        return getattr(torch.ops._moe_C, name)
    except Exception:
        return None


pytestmark = pytest.mark.skipif(
    _moe_op("topk_softmax") is None,
    reason="torch.ops._moe_C.topk_softmax (MetaX MoE gating kernel) is unavailable",
)

# kind -> (op name, activation the kernel applies to the gating logits)
GATERS = {
    "softmax": ("topk_softmax", lambda g: torch.softmax(g, dim=-1)),
    "sigmoid": ("topk_sigmoid", lambda g: g.sigmoid()),
}

# (num_experts, topk)
SHAPES = [
    (8, 2),
    (16, 2),
    (64, 6),
    (128, 8),
    (256, 8),
]
NUM_TOKENS = [1, 7, 64]
SEEDS = [0, 1]

# Known kernel bug: at exactly (num_experts=128, topk=8) the gating kernels take
# a wrong dispatch branch -- topk_sigmoid falls through to the softmax weight
# path, and the bias path returns biased instead of unbiased weights. The
# selected expert IDs are still correct; only the routing weights are wrong. The
# defect is specific to this single (experts, topk) tile (tk=8 is fine at every
# other expert count; 128 experts is fine at every other topk) and is
# data-independent. Tracked as a strict xfail so a future kernel fix flips it
# green and flags the regression here.
ANOMALY_SHAPE = (128, 8)


def _run_kernel(
    op,
    gating: torch.Tensor,
    topk: int,
    renormalize: bool,
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate the in-place outputs, run the kernel, return (weights, ids)."""
    num_tokens = gating.shape[0]
    device = gating.device
    topk_weights = torch.empty((num_tokens, topk), dtype=torch.float32, device=device)
    topk_indices = torch.empty((num_tokens, topk), dtype=torch.int32, device=device)
    # token_expert_indices is auxiliary scratch the kernel fills for the
    # downstream permute step; we don't assert on it here.
    token_expert_indices = torch.empty(
        (num_tokens, topk), dtype=torch.int32, device=device
    )
    op(
        topk_weights,
        topk_indices,
        token_expert_indices,
        gating,
        renormalize,
        bias,
    )
    return topk_weights, topk_indices


def _ref_gate(
    gating: torch.Tensor,
    activation,
    topk: int,
    renormalize: bool,
    bias: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference matching vLLM-metax ``maca_grouped_topk`` / upstream ``torch_topk``.

    With a correction bias, experts are selected on the *biased* scores but the
    routing weights are gathered from the *unbiased* scores -- this asymmetry is
    the documented contract, not an accident.
    """
    scores = activation(gating)
    if bias is not None:
        selection = scores + bias.unsqueeze(0)
        topk_ids = torch.topk(selection, k=topk, dim=-1).indices
        topk_weights = scores.gather(1, topk_ids)  # weights from UNBIASED scores
    else:
        topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def _scatter_weights(
    ids: torch.Tensor, weights: torch.Tensor, num_experts: int
) -> torch.Tensor:
    """Dense [num_token, num_experts] weight map for order-independent compare.

    The kernel and ``torch.topk`` need not return a row's experts in the same
    order; scattering into expert positions removes that dependence while still
    checking which experts were chosen and what weight each got.
    """
    dense = torch.zeros(
        (ids.shape[0], num_experts), dtype=torch.float32, device=ids.device
    )
    dense.scatter_(1, ids.long(), weights.float())
    return dense


@pytest.mark.parametrize("kind", list(GATERS))
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("renormalize", [False, True])
@pytest.mark.parametrize("with_bias", [False, True])
@pytest.mark.parametrize("seed", SEEDS)
@torch.inference_mode()
def test_topk_gate_matches_reference(
    request: pytest.FixtureRequest,
    kind: str,
    shape: tuple[int, int],
    num_tokens: int,
    renormalize: bool,
    with_bias: bool,
    seed: int,
) -> None:
    op_name, activation = GATERS[kind]
    op = _moe_op(op_name)
    if op is None:
        pytest.skip(f"torch.ops._moe_C.{op_name} unavailable")

    num_experts, topk = shape

    # The (128, 8) tile mis-weights for every combination except softmax/no-bias
    # (see ANOMALY_SHAPE). Mark the affected cases as a strict xfail on weights.
    if shape == ANOMALY_SHAPE and not (kind == "softmax" and not with_bias):
        request.node.add_marker(
            pytest.mark.xfail(
                strict=True,
                reason="kernel bug: wrong routing-weight branch at (128 experts, top-8)",
            )
        )

    torch.manual_seed(seed)
    device = "cuda"

    # Continuous random logits => scores are distinct with probability 1, so the
    # top-k selection is unambiguous and tie-break order never matters.
    gating = torch.randn(num_tokens, num_experts, dtype=torch.float32, device=device)
    bias = (
        torch.randn(num_experts, dtype=torch.float32, device=device) * 0.1
        if with_bias
        else None
    )

    # Reference from a pristine copy, in case the kernel scribbles on its input.
    ref_w, ref_ids = _ref_gate(gating.clone(), activation, topk, renormalize, bias)
    k_w, k_ids = _run_kernel(op, gating.contiguous(), topk, renormalize, bias)

    # Same set of experts selected per row (order-independent). The kernel
    # selects correctly even in the anomalous tile, so this is asserted always.
    assert k_ids.shape == ref_ids.shape == (num_tokens, topk)
    torch.testing.assert_close(
        torch.sort(k_ids.long(), dim=-1).values,
        torch.sort(ref_ids.long(), dim=-1).values,
        rtol=0,
        atol=0,
    )

    # Same routing weight on each selected expert. The activation is computed by
    # the kernel in device precision, so allow a small numerical tolerance.
    torch.testing.assert_close(
        _scatter_weights(k_ids, k_w, num_experts),
        _scatter_weights(ref_ids, ref_w, num_experts),
        rtol=1e-3,
        atol=1e-3,
    )


@pytest.mark.parametrize("kind", list(GATERS))
@pytest.mark.parametrize("renormalize", [False, True])
@torch.inference_mode()
def test_topk_gate_renormalize_sums_to_one(kind: str, renormalize: bool) -> None:
    op_name, _ = GATERS[kind]
    op = _moe_op(op_name)
    if op is None:
        pytest.skip(f"torch.ops._moe_C.{op_name} unavailable")

    num_experts, topk, num_tokens = 64, 6, 32
    torch.manual_seed(0)
    gating = torch.randn(num_tokens, num_experts, dtype=torch.float32, device="cuda")

    weights, _ = _run_kernel(op, gating.contiguous(), topk, renormalize, None)
    sums = weights.float().sum(dim=-1)

    if renormalize:
        torch.testing.assert_close(sums, torch.ones_like(sums), rtol=1e-3, atol=1e-3)
    elif kind == "softmax":
        # Un-renormalised softmax weights over a strict expert subset sum to < 1.
        assert torch.all(sums < 1.0 + 1e-3)


@pytest.mark.parametrize("kind", list(GATERS))
@torch.inference_mode()
def test_topk_gate_output_dtypes_and_shapes(kind: str) -> None:
    op_name, _ = GATERS[kind]
    op = _moe_op(op_name)
    if op is None:
        pytest.skip(f"torch.ops._moe_C.{op_name} unavailable")

    num_experts, topk, num_tokens = 64, 6, 5
    torch.manual_seed(0)
    gating = torch.randn(num_tokens, num_experts, dtype=torch.float32, device="cuda")

    weights, ids = _run_kernel(op, gating.contiguous(), topk, True, None)

    assert weights.shape == (num_tokens, topk)
    assert ids.shape == (num_tokens, topk)
    assert weights.dtype == torch.float32
    assert ids.dtype == torch.int32
    # Selected expert ids are in range and distinct within each row.
    assert int(ids.min()) >= 0 and int(ids.max()) < num_experts
    for row in ids.tolist():
        assert len(set(row)) == topk
