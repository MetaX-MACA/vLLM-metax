# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""One fused gather/penalty/scatter over unique sparse logit coordinates."""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _sparse_penalties(
    logits,
    indices,
    parameters,
    num_updates,
    num_rows,
    vocab_size,
    row_stride,
    col_stride,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    active = offsets < num_updates
    rows = tl.load(indices + offsets, active, other=0)
    tokens = tl.load(indices + num_updates + offsets, active, other=0)
    active = (
        active & (rows >= 0) & (rows < num_rows) & (tokens >= 0) & (tokens < vocab_size)
    )
    repetition = tl.load(parameters + offsets, active, other=1.0)
    shift = tl.load(parameters + num_updates + offsets, active, other=0.0)
    ptr = logits + rows * row_stride + tokens * col_stride
    original = tl.load(ptr, active, other=0.0).to(tl.float32)
    updated = (
        tl.where(original > 0, original / repetition, original * repetition) - shift
    )
    tl.store(ptr, updated, active)


def apply_sparse_penalties(
    logits: torch.Tensor, indices: torch.Tensor, parameters: torch.Tensor
) -> torch.Tensor:
    count = indices.shape[1]
    if count:
        _sparse_penalties[(triton.cdiv(count, 256),)](
            logits,
            indices,
            parameters,
            count,
            logits.shape[0],
            logits.shape[1],
            logits.stride(0),
            logits.stride(1),
            BLOCK=256,
        )
    return logits
