# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Fused FP32 Engram gate with a single final cast to the activation dtype."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _engram_gate_kernel(
    X,
    GATE_INPUT,
    KV,
    QW,
    KW,
    O,  # noqa: E741
    D: tl.constexpr,
    HC: tl.constexpr,
    EPS: tl.constexpr,
    CLAMP: tl.constexpr,
    B: tl.constexpr,
    SEPARATE_GATE_INPUT: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // HC
    hc = row % HC
    col = tl.arange(0, B)
    mask = col < D
    x = tl.load(X + row * D + col, mask, 0).to(tl.float32)
    if SEPARATE_GATE_INPUT:
        query = tl.load(GATE_INPUT + row * D + col, mask, 0).to(tl.float32)
    else:
        query = x
    key = tl.load(KV + token * (HC + 1) * D + hc * D + col, mask, 0).to(tl.float32)
    q_weight = tl.load(QW + hc * D + col, mask, 0).to(tl.float32)
    k_weight = tl.load(KW + hc * D + col, mask, 0).to(tl.float32)
    weight = q_weight * k_weight
    rstd = tl.rsqrt(tl.sum(query * query, 0) / D + EPS) * tl.rsqrt(
        tl.sum(key * key, 0) / D + EPS
    )
    dot = tl.sum((query * weight) * key, 0) * rstd * (D**-0.5)
    gate = tl.sigmoid(
        tl.extra.cuda.libdevice.copysign(tl.sqrt(tl.maximum(tl.abs(dot), CLAMP)), dot)
    )
    value = tl.load(KV + token * (HC + 1) * D + HC * D + col, mask, 0).to(tl.float32)
    tl.store(O + row * D + col, x + gate * value, mask)


def fused_engram_gate(
    x: torch.Tensor,
    kv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    clamp_value: float,
    *,
    gate_input: torch.Tensor | None = None,
) -> torch.Tensor:
    query = x if gate_input is None else gate_input
    assert x.ndim == 3 and kv.ndim == 2
    assert query.shape == x.shape
    t, hc, d = x.shape
    assert kv.shape == (t, (hc + 1) * d)
    assert q_weight.shape == k_weight.shape == (hc, d)
    assert all(
        a.is_cuda and a.is_contiguous() and a.device == x.device
        for a in (x, kv, q_weight, k_weight, query)
    )
    assert all(
        a.dtype in (torch.bfloat16, torch.float32)
        for a in (x, kv, q_weight, k_weight, query)
    )
    out = torch.empty_like(x)
    if t:
        _engram_gate_kernel[(t * hc,)](
            x,
            query,
            kv,
            q_weight,
            k_weight,
            out,
            d,
            hc,
            eps,
            clamp_value,
            triton.next_power_of_2(d),
            gate_input is not None,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out
