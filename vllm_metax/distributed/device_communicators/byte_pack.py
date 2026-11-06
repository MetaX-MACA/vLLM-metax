# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Experimental lossless row packing for small MoE dispatches.

No numeric casts, persistent scratch, host synchronization, or communicator state.
Both kernels run on the caller's current stream; the gather callback must order
its output on that stream, as the existing PyNccl all_gatherv contract does.
"""

import math
from collections.abc import Callable, Sequence

import torch
from vllm.triton_utils import tl, triton

_DTYPES = {
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float16,
    torch.bfloat16,
    torch.float32,
    torch.float64,
}


@triton.jit
def _pack_rows(
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    output,
    ROW_BYTES: tl.constexpr,
    WIDTHS: tl.constexpr,
    ITEMS: tl.constexpr,
    SHAPES: tl.constexpr,
    STRIDES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(1)
    byte = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    base: tl.constexpr = 0
    for i in tl.static_range(len(WIDTHS)):
        local = byte - base
        index = local // ITEMS.value[i]
        address = row * STRIDES.value[i][0]
        for j in tl.static_range(len(SHAPES.value[i]) - 1, -1, -1):
            address += index % SHAPES.value[i][j] * STRIDES.value[i][j + 1]
            index //= SHAPES.value[i][j]
        address = address * ITEMS.value[i] + local % ITEMS.value[i]
        if i == 0:
            source = p0
        elif i == 1:
            source = p1
        elif i == 2:
            source = p2
        elif i == 3:
            source = p3
        elif i == 4:
            source = p4
        elif i == 5:
            source = p5
        elif i == 6:
            source = p6
        else:
            source = p7
        source = source.to(tl.pointer_type(tl.uint8))
        mask = (local >= 0) & (local < WIDTHS.value[i])
        value = tl.load(source + address, mask, other=0)
        tl.store(output + row * ROW_BYTES + byte, value, mask)
        base += WIDTHS.value[i]


@triton.jit
def _unpack_rows(
    packed,
    p0,
    p1,
    p2,
    p3,
    p4,
    p5,
    p6,
    p7,
    ROW_BYTES: tl.constexpr,
    WIDTHS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(1)
    byte = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    base: tl.constexpr = 0
    for i in tl.static_range(len(WIDTHS)):
        local = byte - base
        mask = (local >= 0) & (local < WIDTHS.value[i])
        value = tl.load(packed + row * ROW_BYTES + byte, mask, other=0)
        if i == 0:
            destination = p0
        elif i == 1:
            destination = p1
        elif i == 2:
            destination = p2
        elif i == 3:
            destination = p3
        elif i == 4:
            destination = p4
        elif i == 5:
            destination = p5
        elif i == 6:
            destination = p6
        else:
            destination = p7
        destination = destination.to(tl.pointer_type(tl.uint8))
        tl.store(destination + row * WIDTHS.value[i] + local, value, mask)
        base += WIDTHS.value[i]


def metadata_supported(tensors: Sequence[torch.Tensor]) -> bool:
    """Only metadata that all_gatherv already requires to agree across ranks."""
    if not tensors or len(tensors) > 8:
        return False
    return all(
        x.dtype in _DTYPES and 1 <= x.ndim <= 4 and all(d > 0 for d in x.shape[1:])
        for x in tensors
    )


def should_pack(tensors: Sequence[torch.Tensor], sizes: list[int]) -> bool:
    """Conservative experimental envelope, based on HPCC measurements only.

    Metadata must agree across ranks. Do not select a different
    collective schedule using a rank-local layout property or local token count.
    """
    return (
        metadata_supported(tensors)
        and len(sizes) == 4
        and len(set(sizes)) > 1
        and min(sizes) >= 0
        and max(sizes) <= 8
        and sum(math.prod(x.shape[1:]) * x.element_size() for x in tensors) <= 8260
    )


def supported(tensors: Sequence[torch.Tensor]) -> bool:
    """Validate local storage without changing the cross-rank routing decision."""
    if not metadata_supported(tensors):
        return False
    first = tensors[0]
    return all(
        x.layout == torch.strided
        and x.is_cuda
        and x.device == first.device
        and x.shape[0] == first.shape[0]
        and all(s >= 0 for s in x.stride())
        and not x.requires_grad
        and not x.is_conj()
        and not x.is_neg()
        for x in tensors
    )


def pack_rows(tensors: Sequence[torch.Tensor]) -> torch.Tensor:
    if not supported(tensors):
        raise ValueError("Unsupported byte-pack dtype/device/layout or token count")
    widths = tuple(math.prod(x.shape[1:]) * x.element_size() for x in tensors)
    row_bytes = sum(widths)
    output = torch.empty(
        (tensors[0].shape[0], row_bytes),
        dtype=torch.uint8,
        device=tensors[0].device,
    )
    if output.shape[0]:
        _pack_rows[(triton.cdiv(row_bytes, 256), output.shape[0])](
            *tensors,
            *([tensors[0]] * (8 - len(tensors))),
            output,
            row_bytes,
            widths,
            tuple(x.element_size() for x in tensors),
            tuple(tuple(x.shape[1:]) for x in tensors),
            tuple(tuple(x.stride()) for x in tensors),
            256,
        )
    return output


def unpack_rows(
    packed: torch.Tensor, templates: Sequence[torch.Tensor]
) -> list[torch.Tensor]:
    if not metadata_supported(templates):
        raise ValueError("Unsupported byte-pack template dtype, shape or count")
    widths = tuple(math.prod(x.shape[1:]) * x.element_size() for x in templates)
    if packed.dtype != torch.uint8 or packed.ndim != 2 or not packed.is_contiguous():
        raise ValueError("Packed input must be contiguous rank-2 uint8")
    if packed.shape[1] != sum(widths):
        raise ValueError("Packed row width does not match templates")
    outputs = [
        torch.empty(
            (packed.shape[0], *x.shape[1:]),
            dtype=x.dtype,
            device=packed.device,
        )
        for x in templates
    ]
    if packed.shape[0]:
        _unpack_rows[(triton.cdiv(sum(widths), 256), packed.shape[0])](
            packed,
            *outputs,
            *([outputs[0]] * (8 - len(outputs))),
            sum(widths),
            widths,
            256,
        )
    return outputs


def gather_packed(
    tensors: Sequence[torch.Tensor],
    sizes: list[int],
    gather: Callable[..., list[torch.Tensor]],
) -> list[torch.Tensor]:
    """Pack + real collective + unpack, retaining input/output storage ownership."""
    packed = pack_rows(tensors)
    gathered = gather([packed], dim=0, sizes=sizes)
    return unpack_rows(gathered[0], tensors)
