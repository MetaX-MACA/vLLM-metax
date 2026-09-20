# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# Copyright 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd.
"""BF16 indexer fallbacks for the MetaX DeepGEMM attention API.

DeepGEMM 0.2.0+maca0.4.11.1's non-paged BF16 kernel produces incorrect
logits on C500, and its paged wrapper rejects non-contiguous cache pages.
Compute one query against a tile of keys, reducing heads only after ReLU.
Explicit page strides preserve cross-layer cache allocations without copying
the entire cache. This is indexer scoring, not softmax attention.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _mqa_logits(
    Q,
    K,
    W,
    START,
    END,
    BT,
    OUT,
    q_row_stride,
    q_head_stride,
    q_dim_stride,
    k_block_stride,
    k_token_stride,
    k_dim_stride,
    w_row_stride,
    w_head_stride,
    bt_row_stride,
    bt_col_stride,
    end_row_stride,
    end_col_stride,
    N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    NEXT_N: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGED: tl.constexpr,
    LENS_2D: tl.constexpr,
    CLEAN: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr = 32,
    BLOCK_N: tl.constexpr = 64,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    if PAGED:
        req = row // NEXT_N
        step = row % NEXT_N
        start = 0
        if LENS_2D:
            end = tl.load(END + req * end_row_stride + step * end_col_stride)
        else:
            end = tl.load(END + req * end_row_stride) - NEXT_N + step + 1
    else:
        start = tl.load(START + row)
        end = tl.load(END + row)
    valid = (cols < N) & (cols >= start) & (cols < end)
    scores = tl.full((BLOCK_N,), 0, tl.float32)
    if tl.sum(valid.to(tl.int32), 0) > 0:
        dims = tl.arange(0, BLOCK_D)
        if PAGED:
            page = tl.load(
                BT + req * bt_row_stride + (cols // PAGE_SIZE) * bt_col_stride,
                mask=valid,
                other=0,
            )
            key_offsets = page * k_block_stride + (cols % PAGE_SIZE) * k_token_stride
        else:
            key_offsets = cols * k_token_stride
        keys = tl.load(
            K + key_offsets[None, :] + dims[:, None] * k_dim_stride,
            mask=valid[None, :] & (dims[:, None] < D),
            other=0,
        )
        for h_start in range(tl.cdiv(H, BLOCK_H)):
            heads = h_start * BLOCK_H + tl.arange(0, BLOCK_H)
            queries = tl.load(
                Q
                + row * q_row_stride
                + heads[:, None] * q_head_stride
                + dims[None, :] * q_dim_stride,
                mask=(heads[:, None] < H) & (dims[None, :] < D),
                other=0,
            )
            logits = tl.maximum(tl.dot(queries, keys), 0)
            weights = tl.load(
                W + row * w_row_stride + heads * w_head_stride,
                mask=heads < H,
                other=0,
            ).to(tl.float32)
            scores += tl.sum(logits * weights[:, None], axis=0)
    scores = tl.where(valid, scores, float("-inf") if CLEAN else 0.0)
    tl.store(OUT + row * N + cols, scores, mask=cols < N)


def bf16_mqa_logits(
    q: torch.Tensor,
    kv: torch.Tensor,
    weights: torch.Tensor,
    starts: torch.Tensor,
    ends: torch.Tensor,
    *,
    block_table: torch.Tensor | None = None,
    max_model_len: int | None = None,
    clean_logits: bool = True,
) -> torch.Tensor:
    """Score flat KV or paged KV with one causal bound per query.

    Flat: Q [M,H,D], KV [N,D], starts/ends [M].
    Paged: Q [B,S,H,D], KV [pages,page_size,1,D], ends [B] or [B,S].
    Weights are [M,H] or [B*S,H]; no quantization scales are involved.
    """
    paged = block_table is not None
    next_n = q.shape[1] if paged else 1
    heads, dim = q.shape[-2:]
    if q.dtype != torch.bfloat16 or kv.dtype != torch.bfloat16:
        raise ValueError("BF16 indexer logits require BF16 Q and KV")
    q = q.reshape(-1, heads, dim)
    n = max_model_len if paged else kv.shape[0]
    assert n is not None and n >= 0
    out = torch.empty((q.shape[0], n), dtype=torch.float32, device=q.device)
    if not out.numel():
        return out
    _mqa_logits[(q.shape[0], triton.cdiv(n, 64))](
        q,
        kv,
        weights,
        starts,
        ends,
        block_table,
        out,
        *q.stride(),
        kv.stride(0) if paged else 0,
        kv.stride(1) if paged else kv.stride(0),
        kv.stride(-1),
        *weights.stride(),
        block_table.stride(0) if block_table is not None else 0,
        block_table.stride(1) if block_table is not None else 0,
        ends.stride(0),
        ends.stride(1) if ends.ndim == 2 else 0,
        N=n,
        H=heads,
        D=dim,
        NEXT_N=next_n,
        PAGE_SIZE=kv.shape[1] if paged else 1,
        PAGED=paged,
        LENS_2D=ends.ndim == 2,
        CLEAN=clean_logits,
        BLOCK_D=triton.next_power_of_2(dim),
        num_warps=4,
        num_stages=1,
    )
    return out
