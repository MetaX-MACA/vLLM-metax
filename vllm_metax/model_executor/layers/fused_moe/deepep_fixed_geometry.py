# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Keep a single native LL expert geometry while preserving rank-local experts."""

import os
import torch
from vllm.triton_utils import triton, tl


def wire_experts(buffer, actual):
    if os.getenv("VLLM_METAX_DEEPEP_LL_FIXED_EXPERT_GEOMETRY", "0") != "1":
        return actual
    capacity = int(os.environ["VLLM_METAX_DEEPEP_LL_BUFFER_MAX_EXPERTS"])
    world = buffer.group_size
    if actual > capacity or actual % world or capacity % world:
        raise ValueError("Incompatible logical/native DeepEP expert geometry")
    return capacity


def map_expert_ids(ids, actual, wire, world):
    if actual == wire:
        return ids
    local = actual // world
    wire_local = wire // world
    return torch.where(
        ids >= 0,
        torch.div(ids, local, rounding_mode="floor") * wire_local + ids % local,
        ids,
    )


@triton.jit
def _copy_valid(
    dst,
    src,
    counts,
    S0: tl.constexpr,
    S1: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    H: tl.constexpr,
    B: tl.constexpr,
):
    expert = tl.program_id(0)
    cols = tl.program_id(1) * B + tl.arange(0, B)
    rows = tl.load(counts + expert)
    for row in range(rows):
        values = tl.load(src + expert * S0 + row * S1 + cols, cols < H, other=0)
        tl.store(dst + expert * D0 + row * D1 + cols, values, cols < H)


def copy_valid_rows(dst, src, counts):
    assert dst.dtype == src.dtype == torch.bfloat16
    assert src.ndim == dst.ndim == 3 and src.shape[0] == counts.numel()
    assert (
        dst.shape[0] >= src.shape[0]
        and dst.shape[1] >= src.shape[1]
        and dst.shape[2] == src.shape[2]
    )
    assert src.stride(2) == dst.stride(2) == 1
    _copy_valid[(src.shape[0], triton.cdiv(src.shape[2], 1024))](
        dst,
        src,
        counts,
        src.stride(0),
        src.stride(1),
        dst.stride(0),
        dst.stride(1),
        src.shape[2],
        1024,
    )
