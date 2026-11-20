# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Pack native INT8 quantization results without changing their numeric rules."""

import torch
from vllm.triton_utils import tl, triton
from vllm_metax.model_executor.layers.fused_moe import source_packed_dispatch as packed


@triton.jit
def _quant_and_pack_source(
    X,
    OUT,
    K: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask_k = cols < K
    x = tl.load(X + row * K + cols, mask=mask_k, other=0.0).to(tl.float32)

    absmax = tl.max(tl.abs(x), 0)
    scale = tl.div_rn(absmax, 127.0)
    inv = tl.where(absmax == 0.0, 0.0, tl.div_rn(127.0, absmax))

    q_f = tl.extra.cuda.libdevice.nearbyint(x * inv)
    q = tl.minimum(tl.maximum(q_f, -128.0), 127.0).to(tl.int8)

    # 1. Store quantized INT8 elements
    tl.store(OUT + row * WIDTH + cols, q, mask=mask_k)

    # 2. Store 4 scale bytes
    byte = tl.arange(0, 4)
    bits = scale.to(tl.uint32, bitcast=True)
    scale_byte = ((bits >> (byte * 8)) & 255).to(tl.uint8)
    tl.store(OUT + row * WIDTH + K + byte, scale_byte)


def quant_and_pack_source(hidden: torch.Tensor) -> torch.Tensor:
    """Fused single-kernel operator: quantize BF16 hidden states directly into packed carrier.

    Combines dynamic per-token absmax/scale calculation, nearbyint INT8 quantization,
    scale-byte packing and zero-padding into a single memory pass without intermediate
    q/scale HBM allocations or round-trips.
    """
    assert (
        hidden.ndim == 2 and hidden.dtype == torch.bfloat16 and hidden.is_contiguous()
    )
    rows, k = hidden.shape
    width = triton.cdiv(k + 4, 512) * 512
    carrier = torch.zeros(
        (rows, width // 2), dtype=torch.bfloat16, device=hidden.device
    )
    if rows:
        _quant_and_pack_source[(rows,)](
            hidden,
            carrier.view(torch.int8),
            k,
            width,
            triton.next_power_of_2(width),
            num_warps=4,
        )
    return carrier


def pack_hidden(hidden: torch.Tensor) -> torch.Tensor:
    if hidden.ndim != 2 or hidden.dtype != torch.bfloat16 or not hidden.is_contiguous():
        raise ValueError("W4A8 source packing requires contiguous 2D BF16 normed input")
    if hidden.shape[0] == 0:
        return hidden.new_empty((0, packed.carrier_width(hidden.shape[1])))
    return quant_and_pack_source(hidden)


def pack_hidden_fake(hidden: torch.Tensor) -> torch.Tensor:
    return hidden.new_empty((hidden.shape[0], packed.carrier_width(hidden.shape[1])))
