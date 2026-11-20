# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""INT8 DeepEP-LL helpers that visit only GPU-reported valid expert rows.

Padding is intentionally undefined. Consumers MUST be masked GEMM/DeepEP
combine; these buffers are not suitable for dense activation/quant kernels.
No expert counts are copied to the CPU, including during graph capture.
"""

import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.triton_utils import tl, triton

from vllm_metax import envs as mx_envs


def enabled() -> bool:
    return mx_envs.VLLM_METAX_MASKED_INT8_MOE


class PackedInt8TokensMetadata(mk.ExpertTokensMetadata):
    """Normal expert counts plus the packed input produced by the receiver."""

    def __init__(
        self, expert_num_tokens: torch.Tensor, packed_input: torch.Tensor | None = None
    ):
        super().__init__(
            expert_num_tokens=expert_num_tokens, expert_num_tokens_cpu=None
        )
        self.packed_input = packed_input


@triton.jit
def _quant_row(x):
    absmax = tl.max(tl.abs(x), 0)
    scale = tl.div_rn(absmax, 127.0)
    inv = tl.where(absmax == 0.0, 0.0, tl.div_rn(127.0, absmax))
    q = tl.extra.cuda.libdevice.nearbyint(x * inv)
    return tl.minimum(tl.maximum(q, -128.0), 127.0).to(tl.int8), scale


@triton.jit
def _store_packed_row(PACKED, index, q, scale, cols, K: tl.constexpr):
    """Row layout: K int8 values followed by the float32 scale as 4 bytes."""
    tl.store(PACKED + index * (K + 4) + cols, q, cols < K)
    byte = tl.arange(0, 4)
    bits = scale.to(tl.uint32, bitcast=True)
    scale_byte = ((bits >> (byte * 8)) & 255).to(tl.uint8)
    tl.store(PACKED + index * (K + 4) + K + byte, scale_byte)


@triton.jit
def _quant_valid_rows(
    X,
    Q,
    S,
    PACKED,
    COUNTS,
    T: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    LANES: tl.constexpr,
):
    expert = tl.program_id(0)
    lane = tl.program_id(1)
    count = tl.load(COUNTS + expert)
    cols = tl.arange(0, BLOCK)
    for row in range(lane, count, LANES):
        offset = (expert * T + row) * K
        x = tl.load(X + offset + cols, cols < K, 0).to(tl.float32)
        q, scale = _quant_row(x)
        tl.store(Q + offset + cols, q, cols < K)
        tl.store(S + expert * T + row, scale)
        _store_packed_row(PACKED, expert * T + row, q, scale, cols, K)


@triton.jit
def _pack_valid_rows(
    Q,
    S,
    PACKED,
    COUNTS,
    T: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    LANES: tl.constexpr,
):
    expert = tl.program_id(0)
    lane = tl.program_id(1)
    count = tl.load(COUNTS + expert)
    cols = tl.arange(0, BLOCK)
    for row in range(lane, count, LANES):
        index = expert * T + row
        q = tl.load(Q + index * K + cols, cols < K, 0)
        _store_packed_row(PACKED, index, q, tl.load(S + index), cols, K)


def quantize_and_pack_valid_rows(x: torch.Tensor, counts: torch.Tensor):
    assert x.ndim == 3 and x.is_contiguous() and x.dtype == torch.bfloat16
    e, t, k = x.shape
    # Retain the normal q/scale interface for modular-kernel shape validation
    # and fallback experts; the packed copy is written by the same launch.
    q = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    scale = torch.empty((e, t, 1), dtype=torch.float32, device=x.device)
    packed = torch.empty((e, t, k + 4), dtype=torch.int8, device=x.device)
    _quant_valid_rows[(e, 4)](
        x, q, scale, packed, counts, t, k, triton.next_power_of_2(k), 4, num_warps=4
    )
    return q, scale, packed


def pack_valid_rows(
    q: torch.Tensor, scale: torch.Tensor, counts: torch.Tensor
) -> torch.Tensor:
    e, t, k = q.shape
    packed = torch.empty((e, t, k + 4), dtype=torch.int8, device=q.device)
    _pack_valid_rows[(e, 4)](
        q, scale, packed, counts, t, k, triton.next_power_of_2(k), 4, num_warps=4
    )
    return packed


@triton.jit
def _silu_quant_pack(
    X,
    PACKED,
    COUNTS,
    T: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    LANES: tl.constexpr,
):
    expert = tl.program_id(0)
    lane = tl.program_id(1)
    count = tl.load(COUNTS + expert)
    cols = tl.arange(0, BLOCK)
    for row in range(lane, count, LANES):
        index = expert * T + row
        gate = tl.load(X + index * (2 * K) + cols, cols < K, 0).to(tl.float32)
        up = tl.load(X + index * (2 * K) + K + cols, cols < K, 0).to(tl.float32)
        # The installed MACA op rounds SiLU to BF16 before multiplying by up,
        # then rounds the product again. Preserve BOTH rounding boundaries.
        silu = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
        a = (silu * up).to(tl.bfloat16).to(tl.float32)
        q, scale = _quant_row(a)
        _store_packed_row(PACKED, index, q, scale, cols, K)


def supports_clamp(clamp_limit: float | None) -> bool:
    """Clamped SwiGLU (swiglu_limit) is only implemented by the native masked op."""
    return clamp_limit is None or (
        clamp_limit > 0 and mx_envs.VLLM_METAX_NATIVE_MASKED_ACT
    )


def activation_route(activation, limit, alpha, beta):
    """Pick the fused-op variant for gate*sigmoid(alpha*gate)*(up+beta) (clamped), or None.
    'silu': alpha == 1, beta == 0 (DeepSeek-V4 style: silu(gate)*up, optional clamp).
    'oai' : alpha > 0, beta == 1, limit > 0 (gpt-oss style, packed gate|up layout), the same
            variant SGLang selects with gemm1_alpha > 0; the op hard-codes the (up + 1).
    vLLM's unset alpha/beta are 1.0/0.0 (SGLang's unset alpha is 0.0), so they must not be
    mistaken for an explicit alpha. Other combinations are not expressible by the op.
    """
    a = 1.0 if alpha is None else float(alpha)
    b = 0.0 if beta is None else float(beta)
    name = getattr(activation, "value", activation)
    if name == "silu" and a == 1.0 and b == 0.0:
        return "silu" if supports_clamp(limit) else None
    if (
        name == "swigluoai_uninterleave"
        and a > 0
        and b == 1.0
        and limit is not None
        and limit > 0
        and mx_envs.VLLM_METAX_NATIVE_MASKED_ACT
    ):
        return "oai"
    return None


def silu_quant_pack_valid_rows(
    x: torch.Tensor, counts: torch.Tensor, limit=None, route="silu", alpha=1.0, beta=0.0
):
    if mx_envs.VLLM_METAX_NATIVE_MASKED_ACT:
        return native_activation(x, counts, limit, route, alpha, beta)
    assert limit is None and route == "silu", (
        "clamped/OAI SwiGLU needs VLLM_METAX_NATIVE_MASKED_ACT"
    )
    e, t, n = x.shape
    assert n % 2 == 0 and x.dtype == torch.bfloat16
    k = n // 2
    packed = torch.empty((e, t, k + 4), dtype=torch.int8, device=x.device)
    _silu_quant_pack[(e, 4)](
        x,
        packed,
        counts,
        t,
        k,
        triton.next_power_of_2(k),
        4,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return packed


def native_activation(
    x: torch.Tensor,
    counts: torch.Tensor,
    limit=None,
    route="silu",
    alpha=None,
    beta=0.0,
) -> torch.Tensor:
    """SGL's native masked SiLU/INT8/pack layout, passed whole to GEMM2."""
    from mcoplib import op

    assert x.ndim == 3 and x.dtype == torch.bfloat16 and x.is_contiguous()
    k = x.shape[-1] // 2
    # Same layout as sglang ep_moe/kernels.py::silu_and_mul_masked.
    width = ((k // 2 + 257) // 256) * 256
    out = x.new_empty((x.shape[0], x.shape[1], width))
    # 4th arg is swiglu_limit: gate<=limit, -limit<=up<=limit (0.0 = no clamp)
    if route == "oai":
        op.fused_silu_mul_dq_mask_quant(
            out, x, counts, gemm1_alpha=float(alpha), gemm1_limit=float(limit)
        )
    else:
        op.fused_silu_mul_dq_mask_quant(
            out, x, counts, 0.0 if limit is None else float(limit)
        )
    # Preserve padded row pitch. Never slice [:K+4] into a strided view.
    return out.view(torch.int8)


@triton.jit
def _compact(
    X,
    OUT,
    COUNTS,
    S0: tl.constexpr,
    S1: tl.constexpr,
    T: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    expert, lane = tl.program_id(0), tl.program_id(1)
    count = tl.load(COUNTS + expert)
    cols = tl.arange(0, BLOCK)
    for row in range(lane, count, 4):
        value = tl.load(X + expert * S0 + row * S1 + cols, cols < WIDTH, 0)
        tl.store(OUT + (expert * T + row) * WIDTH + cols, value, cols < WIDTH)


_compact_scratch: dict[tuple[tuple[int, int, int], torch.device], torch.Tensor] = {}


def compact_carrier(carrier, counts, k):
    raw = carrier.view(torch.int8)
    assert raw.ndim == 3 and raw.stride(2) == 1 and raw.shape[2] >= k + 4
    if raw.shape[2] == k + 4 and raw.is_contiguous():
        return raw
    e, t, _ = raw.shape
    shape = (e, t, k + 4)
    key = (shape, raw.device)
    out = _compact_scratch.get(key)
    if out is None or out.shape != shape:
        out = _compact_scratch[key] = torch.empty(
            shape, device=raw.device, dtype=torch.int8
        )
    _compact[(e, 4)](
        raw,
        out,
        counts,
        raw.stride(0),
        raw.stride(1),
        t,
        k + 4,
        triton.next_power_of_2(k + 4),
        num_warps=4,
    )
    return out


@triton.jit
def _silu_quant_pack_compact(
    X,
    PACKED,
    COUNTS,
    T: tl.constexpr,
    K: tl.constexpr,
    LIMIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    expert, lane = tl.program_id(0), tl.program_id(1)
    count = tl.load(COUNTS + expert)
    cols = tl.arange(0, BLOCK)
    for row in range(lane, count, 4):
        index = expert * T + row
        gate = tl.load(X + index * (2 * K) + cols, cols < K, 0).to(tl.float32)
        up = tl.load(X + index * (2 * K) + K + cols, cols < K, 0).to(tl.float32)
        if LIMIT > 0:
            gate = tl.minimum(gate, LIMIT)
            up = tl.minimum(tl.maximum(up, -LIMIT), LIMIT)
            act = (gate / (1.0 + tl.exp(-gate)) * up).to(tl.bfloat16).to(tl.float32)
        else:
            silu = (gate / (1.0 + tl.exp(-gate))).to(tl.bfloat16).to(tl.float32)
            act = (silu * up).to(tl.bfloat16).to(tl.float32)
        q, scale = _quant_row(act)
        _store_packed_row(PACKED, index, q, scale, cols, K)


def supports_compact_activation(activation, config):
    """Keep the INT4 consumer's activation predicate, including clamp_limit=0."""
    name = getattr(activation, "value", activation)
    return (
        name == "silu"
        and (config.alpha is None or config.alpha == 1.0)
        and (config.beta is None or config.beta == 0.0)
        and (config.clamp_limit is None or config.clamp_limit >= 0)
    )


def activation_and_pack_compact(x, counts, config):
    """Produce logical K+4 rows for consumers that reject padded row widths.

    Preserve the original clamped Triton rounding and native compaction. The
    padded W8A8 consumer keeps silu_quant_pack_valid_rows and its own predicate.
    """
    k = x.shape[-1] // 2
    if mx_envs.VLLM_METAX_NATIVE_MASKED_ACT:
        # The native INT8 carrier is 512B aligned. The W4A8 Python wrapper
        # requires logical K+4, so compact valid rows rather than slicing a
        # strided prefix or changing DeepGEMM's K checks.
        native = native_activation(x, counts, config.clamp_limit)
        return compact_carrier(native, counts, k)
    e, t, _ = x.shape
    packed = torch.empty((e, t, k + 4), device=x.device, dtype=torch.int8)
    _silu_quant_pack_compact[(e, 4)](
        x,
        packed,
        counts,
        t,
        k,
        0.0 if config.clamp_limit is None else float(config.clamp_limit),
        triton.next_power_of_2(k),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return packed


def pack_per_token_int8(x, scales):
    """Native per-token INT4 GEMM expects the FP32 scale after each INT8 row."""
    e, m, k = x.shape
    if scales is None or scales.numel() != e * m:
        raise ValueError("W4A8 masked GEMM requires one activation scale per row")
    scale_bytes = (
        scales.to(torch.float32).contiguous().reshape(e, m, 1).view(torch.int8)
    )
    return torch.cat((x, scale_bytes), dim=-1).contiguous()
