# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for Qwen4Exp QSA sparse attention and cache updates."""

from __future__ import annotations

import os

import torch

try:
    # This is the production import path.  Keeping it first means this file
    # can replace ``metax/ops/qsa.py`` without any source changes.
    from vllm.model_executor.warmup.jit_warmup_triton_helper import (
        TritonWarmupTensor,
        triton_scalar_specialization_rep,
    )
    from vllm.triton_utils import HAS_TRITON, tl, triton
except ModuleNotFoundError:
    # The microbench package intentionally does not contain vLLM's warmup
    # helpers.  This fallback only makes the same production source importable
    # for standalone Triton correctness and benchmark runs.
    import triton
    import triton.language as tl

    HAS_TRITON = True

    class TritonWarmupTensor:  # pragma: no cover - microbench never warms up
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("Triton warmup requires the vLLM runtime")

    def triton_scalar_specialization_rep(value: int) -> int:
        return value


@triton.jit(do_not_specialize=["num_rows", "num_requests"])
def _qsa_sparse_paged_fma_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    indices_ptr,
    table_ptr,
    req_ptr,
    partial_ptr,
    lse_ptr,
    out_ptr,
    sqr,
    sqh,
    skb,
    skt,
    skh,
    svb,
    svt,
    svh,
    sir,
    str_,
    sor,
    soh,
    num_rows,
    cache_blocks,
    num_requests,
    TOPK: tl.constexpr,
    PAGE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    DIM: tl.constexpr,
    SPLITS: tl.constexpr,
    TILES: tl.constexpr,
    BN: tl.constexpr,
    GROUPS: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    GQA_GROUP_SIZE: tl.constexpr,
    HEAD_GROUPS: tl.constexpr,
    HEADS_PER_STREAM: tl.constexpr,
    SHARED_KV_PAGE_STRIDE: tl.constexpr,
    GROUPED_TOKENS: tl.constexpr,
):
    """FMA backend serving query heads in groups of three.

    QSA expansion makes each group four adjacent logical tokens.  Their
    page-table entry is identical, so lookup happens once per group while K/V
    values remain token granular and the final causal group remains masked.
    A masked tail lets the same kernel cover every GQA group size.
    """
    row = tl.program_id(0)
    head_stream = tl.program_id(1)
    split = tl.program_id(2)
    kv_head = head_stream // HEAD_GROUPS
    head_group = head_stream % HEAD_GROUPS
    head0 = kv_head * GQA_GROUP_SIZE + head_group * HEADS_PER_STREAM
    req = tl.load(req_ptr + row)
    safe_req = tl.minimum(tl.maximum(req, 0), num_requests - 1)
    count = tl.minimum(TOPK, tl.maximum(0, tl.load(indices_ptr + row * sir + TOPK)))
    dim = tl.arange(0, DIM)
    group_offset = tl.arange(0, GROUPS)
    within_group = tl.arange(0, 4)
    query0 = tl.load(q_ptr + row * sqr + head0 * sqh + dim).to(tl.float32)
    maximum0 = tl.full((), -1.0e20, tl.float32)
    maximum1 = tl.full((), -1.0e20, tl.float32)
    maximum2 = tl.full((), -1.0e20, tl.float32)
    denom0 = tl.full((), 0.0, tl.float32)
    denom1 = tl.full((), 0.0, tl.float32)
    denom2 = tl.full((), 0.0, tl.float32)
    acc0 = tl.zeros((DIM,), tl.float32)
    acc1 = tl.zeros((DIM,), tl.float32)
    acc2 = tl.zeros((DIM,), tl.float32)
    if HEADS_PER_STREAM > 1:
        query1 = tl.load(q_ptr + row * sqr + (head0 + 1) * sqh + dim).to(tl.float32)
    if HEADS_PER_STREAM > 2:
        query2 = tl.load(q_ptr + row * sqr + (head0 + 2) * sqh + dim).to(tl.float32)
    scale: tl.constexpr = (DIM**-0.5) * 1.4426950408889634
    tile_end = tl.minimum(TILES, tl.cdiv(count, BN))
    # Seed the first tile. Each iteration prepares the next tile's short
    # selection/page-table dependency chain before consuming current K/V.
    # This spends a few registers to overlap metadata latency without keeping
    # a second payload tile live.
    group_columns = split * BN + group_offset * 4
    columns = group_columns[:, None] + within_group[None, :]
    if GROUPED_TOKENS:
        token_base = tl.load(
            indices_ptr + row * sir + group_columns,
            mask=group_columns < count,
            other=-1,
        )
        safe_base = tl.maximum(token_base, 0)
        page = safe_base // PAGE
        base_offset = safe_base % PAGE
        group_valid = (
            (group_columns < count)
            & (token_base >= 0)
            & (req >= 0)
            & (req < num_requests)
            & (page < TABLE_WIDTH)
        )
        physical = tl.load(
            table_ptr + safe_req * str_ + tl.minimum(page, TABLE_WIDTH - 1),
            mask=group_valid,
            other=-1,
        )
        group_valid &= (physical >= 0) & (physical < cache_blocks)
        valid = group_valid[:, None] & (columns < count)
        # Preserve the one-dimensional page/offset layout for grouped tokens.
        # Expanding these before address generation makes the C500 backend
        # rebuild the gather indices in a slower two-dimensional layout.
        safe_physical = tl.maximum(physical, 0).to(tl.int64)
        token_offset = base_offset
    else:
        token = tl.load(
            indices_ptr + row * sir + columns,
            mask=columns < count,
            other=-1,
        )
        safe_token = tl.maximum(token, 0)
        page = safe_token // PAGE
        token_offset = safe_token % PAGE
        valid = (
            (columns < count)
            & (token >= 0)
            & (req >= 0)
            & (req < num_requests)
            & (page < TABLE_WIDTH)
        )
        physical = tl.load(
            table_ptr + safe_req * str_ + tl.minimum(page, TABLE_WIDTH - 1),
            mask=valid,
            other=-1,
        )
        valid &= (physical >= 0) & (physical < cache_blocks)
        safe_physical = tl.maximum(physical, 0).to(tl.int64)
    # The physical page must be widened before multiplying the block stride;
    # a fully int32 paged address can overflow for large caches.
    key_page_offset = safe_physical * skb
    if SHARED_KV_PAGE_STRIDE:
        value_page_offset = key_page_offset
    else:
        value_page_offset = safe_physical * svb
    for tile in tl.range(split, tile_end, SPLITS):
        next_group_columns = (tile + SPLITS) * BN + group_offset * 4
        next_columns = next_group_columns[:, None] + within_group[None, :]
        if GROUPED_TOKENS:
            next_token_base = tl.load(
                indices_ptr + row * sir + next_group_columns,
                mask=next_group_columns < count,
                other=-1,
            )
            next_safe_base = tl.maximum(next_token_base, 0)
            next_page = next_safe_base // PAGE
            next_base_offset = next_safe_base % PAGE
            next_group_valid = (
                (next_group_columns < count)
                & (next_token_base >= 0)
                & (req >= 0)
                & (req < num_requests)
                & (next_page < TABLE_WIDTH)
            )
            next_physical = tl.load(
                table_ptr + safe_req * str_ + tl.minimum(next_page, TABLE_WIDTH - 1),
                mask=next_group_valid,
                other=-1,
            )
            next_group_valid &= (next_physical >= 0) & (next_physical < cache_blocks)
            next_valid = next_group_valid[:, None] & (next_columns < count)
            next_safe_physical = tl.maximum(next_physical, 0).to(tl.int64)
            next_token_offset = next_base_offset
        else:
            next_token = tl.load(
                indices_ptr + row * sir + next_columns,
                mask=next_columns < count,
                other=-1,
            )
            next_safe_token = tl.maximum(next_token, 0)
            next_page = next_safe_token // PAGE
            next_token_offset = next_safe_token % PAGE
            next_valid = (
                (next_columns < count)
                & (next_token >= 0)
                & (req >= 0)
                & (req < num_requests)
                & (next_page < TABLE_WIDTH)
            )
            next_physical = tl.load(
                table_ptr + safe_req * str_ + tl.minimum(next_page, TABLE_WIDTH - 1),
                mask=next_valid,
                other=-1,
            )
            next_valid &= (next_physical >= 0) & (next_physical < cache_blocks)
            next_safe_physical = tl.maximum(next_physical, 0).to(tl.int64)
        next_key_page_offset = next_safe_physical * skb
        if SHARED_KV_PAGE_STRIDE:
            next_value_page_offset = next_key_page_offset
        else:
            next_value_page_offset = next_safe_physical * svb
        if GROUPED_TOKENS:
            keys = tl.load(
                k_ptr
                + key_page_offset[:, None, None]
                + (token_offset[:, None, None] + within_group[None, :, None]) * skt
                + kv_head * skh
                + dim[None, None, :],
                mask=valid[:, :, None],
                other=0,
            ).to(tl.float32)
        else:
            keys = tl.load(
                k_ptr
                + key_page_offset[:, :, None]
                + token_offset[:, :, None] * skt
                + kv_head * skh
                + dim[None, None, :],
                mask=valid[:, :, None],
                other=0,
            ).to(tl.float32)
        # Keep metadata, masks and payload in the same [group, token, dim]
        # layout. Flattening before the reductions makes the C500 backend
        # rebuild the index and page-table path in a second linear layout.
        scores0 = tl.where(
            valid,
            tl.sum(keys * query0[None, None, :], axis=2) * scale,
            -1.0e20,
        )
        if HEADS_PER_STREAM > 1:
            scores1 = tl.where(
                valid,
                tl.sum(keys * query1[None, None, :], axis=2) * scale,
                -1.0e20,
            )
        if HEADS_PER_STREAM > 2:
            scores2 = tl.where(
                valid,
                tl.sum(keys * query2[None, None, :], axis=2) * scale,
                -1.0e20,
            )
        if GROUPED_TOKENS:
            values = tl.load(
                v_ptr
                + value_page_offset[:, None, None]
                + (token_offset[:, None, None] + within_group[None, :, None]) * svt
                + kv_head * svh
                + dim[None, None, :],
                mask=valid[:, :, None],
                other=0,
            ).to(tl.float32)
        else:
            values = tl.load(
                v_ptr
                + value_page_offset[:, :, None]
                + token_offset[:, :, None] * svt
                + kv_head * svh
                + dim[None, None, :],
                mask=valid[:, :, None],
                other=0,
            ).to(tl.float32)
        # Update each head before creating the next probability vector. This
        # shortens temporary live ranges on C500 while keeping K/V shared.
        next_max0 = tl.maximum(maximum0, tl.max(tl.max(scores0, axis=1), axis=0))
        probs0 = tl.where(valid, tl.math.exp2(scores0 - next_max0), 0)
        weighted0 = tl.sum(probs0[:, :, None] * values, axis=1)
        acc0 = acc0 * tl.math.exp2(maximum0 - next_max0) + tl.sum(weighted0, axis=0)
        denom0 = denom0 * tl.math.exp2(maximum0 - next_max0) + tl.sum(
            tl.sum(probs0, axis=1), axis=0
        )
        if HEADS_PER_STREAM > 1:
            next_max1 = tl.maximum(maximum1, tl.max(tl.max(scores1, axis=1), axis=0))
            probs1 = tl.where(valid, tl.math.exp2(scores1 - next_max1), 0)
            weighted1 = tl.sum(probs1[:, :, None] * values, axis=1)
            acc1 = acc1 * tl.math.exp2(maximum1 - next_max1) + tl.sum(weighted1, axis=0)
            denom1 = denom1 * tl.math.exp2(maximum1 - next_max1) + tl.sum(
                tl.sum(probs1, axis=1), axis=0
            )
        if HEADS_PER_STREAM > 2:
            next_max2 = tl.maximum(maximum2, tl.max(tl.max(scores2, axis=1), axis=0))
            probs2 = tl.where(valid, tl.math.exp2(scores2 - next_max2), 0)
            weighted2 = tl.sum(probs2[:, :, None] * values, axis=1)
            acc2 = acc2 * tl.math.exp2(maximum2 - next_max2) + tl.sum(weighted2, axis=0)
            denom2 = denom2 * tl.math.exp2(maximum2 - next_max2) + tl.sum(
                tl.sum(probs2, axis=1), axis=0
            )
        maximum0 = next_max0
        if HEADS_PER_STREAM > 1:
            maximum1 = next_max1
        if HEADS_PER_STREAM > 2:
            maximum2 = next_max2
        token_offset = next_token_offset
        valid = next_valid
        key_page_offset = next_key_page_offset
        value_page_offset = next_value_page_offset
    partial_row = (split * num_rows + row).to(tl.int64)
    for head in range(0, HEADS_PER_STREAM):
        if head == 0:
            maximum, denom, acc = maximum0, denom0, acc0
        elif head == 1:
            maximum, denom, acc = maximum1, denom1, acc1
        else:
            maximum, denom, acc = maximum2, denom2, acc2
        has_values = denom > 0
        result = tl.where(has_values, acc / tl.maximum(denom, 1.0e-20), 0)
        output_head = head0 + head
        if SPLITS == 1:
            tl.store(
                out_ptr + row * sor + output_head * soh + dim,
                result,
            )
        else:
            tl.store(
                partial_ptr + (partial_row * NUM_QUERY_HEADS + output_head) * DIM + dim,
                result,
            )
            lse = tl.where(
                has_values,
                maximum + tl.math.log2(tl.maximum(denom, 1.0e-20)),
                -float("inf"),
            )
            tl.store(
                lse_ptr + partial_row * NUM_QUERY_HEADS + output_head,
                lse,
            )


@triton.jit
def _qsa_sparse_paged_mma_reverse_tile(
    q_ptr,
    k_ptr,
    v_ptr,
    indices_ptr,
    table_ptr,
    query,
    accumulator,
    denom,
    maximum,
    row,
    kv_head,
    req,
    safe_req,
    count,
    tile,
    skb,
    skt,
    skh,
    svb,
    svt,
    svh,
    sir,
    str_,
    cache_blocks,
    num_requests,
    PAGE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    DIM: tl.constexpr,
    MASK_COUNT: tl.constexpr,
    TRANSPOSE_QK: tl.constexpr,
):
    group = tl.arange(0, 4)
    within_group = tl.arange(0, 4)
    dim = tl.arange(0, DIM)
    group_columns = tile * 16 + group * 4
    if MASK_COUNT:
        token_base = tl.load(
            indices_ptr + row * sir + group_columns,
            mask=group_columns < count,
            other=-1,
        )
    else:
        token_base = tl.load(indices_ptr + row * sir + group_columns)
    safe_base = tl.maximum(token_base, 0)
    logical_page = safe_base // PAGE
    page_offset = safe_base % PAGE
    group_valid = (
        (token_base >= 0)
        & (req >= 0)
        & (req < num_requests)
        & (logical_page < TABLE_WIDTH)
    )
    if MASK_COUNT:
        group_valid &= group_columns < count
    physical = tl.load(
        table_ptr + safe_req * str_ + tl.minimum(logical_page, TABLE_WIDTH - 1),
        mask=group_valid,
        other=-1,
    )
    group_valid &= (physical >= 0) & (physical < cache_blocks)
    if MASK_COUNT:
        valid_2d = group_valid[:, None] & (
            group_columns[:, None] + within_group[None, :] < count
        )
    else:
        valid_2d = group_valid[:, None] & (within_group[None, :] < 4)
    valid = tl.reshape(valid_2d, (16,))
    safe_physical = tl.maximum(physical, 0).to(tl.int64)
    key_page_base = safe_physical * skb
    value_page_base = safe_physical * svb
    keys = tl.load(
        k_ptr
        + key_page_base[:, None, None]
        + (page_offset[:, None, None] + within_group[None, :, None]) * skt
        + kv_head * skh
        + dim[None, None, :],
        mask=valid_2d[:, :, None],
        other=0.0,
    )
    keys = tl.reshape(keys, (16, DIM))
    scale: tl.constexpr = (DIM**-0.5) * 1.4426950408889634
    if TRANSPOSE_QK:
        scores = tl.trans(tl.dot(keys, tl.trans(query))) * scale
    else:
        scores = tl.dot(query, tl.trans(keys)) * scale
    values = tl.load(
        v_ptr
        + value_page_base[:, None, None]
        + (page_offset[:, None, None] + within_group[None, :, None]) * svt
        + kv_head * svh
        + dim[None, None, :],
        mask=valid_2d[:, :, None],
        other=0.0,
    )
    values = tl.reshape(values, (16, DIM))
    scores = tl.where(valid[None, :], scores, -1.0e20)
    next_max = tl.maximum(maximum, tl.max(scores, axis=1))
    alpha = tl.math.exp2(maximum - next_max)
    probabilities = tl.where(
        valid[None, :],
        tl.math.exp2(scores - next_max[:, None]),
        0.0,
    )
    accumulator = tl.trans(
        tl.dot(
            tl.trans(values),
            tl.trans(probabilities.to(values.dtype)),
            acc=tl.trans(accumulator * alpha[:, None]),
        )
    )
    denom = denom * alpha + tl.sum(probabilities, axis=1)
    return accumulator, denom, next_max


@triton.jit
def _qsa_sparse_paged_mma_reverse_two_tiles(
    k_ptr,
    v_ptr,
    indices_ptr,
    table_ptr,
    query,
    accumulator,
    denom,
    maximum,
    row,
    kv_head,
    req,
    safe_req,
    tile0,
    tile1,
    tile1_active,
    skb,
    skt,
    skh,
    svb,
    svt,
    svh,
    sir,
    str_,
    cache_blocks,
    num_requests,
    PAGE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    DIM: tl.constexpr,
):
    """Consume two complete tiles with one online-softmax state update."""
    group = tl.arange(0, 4)
    within_group = tl.arange(0, 4)
    dim = tl.arange(0, DIM)
    columns0 = tile0 * 16 + group * 4
    columns1 = tile1 * 16 + group * 4
    token0 = tl.load(indices_ptr + row * sir + columns0)
    token1 = tl.load(
        indices_ptr + row * sir + columns1,
        mask=tile1_active,
        other=-1,
    )
    safe0 = tl.maximum(token0, 0)
    safe1 = tl.maximum(token1, 0)
    page0 = safe0 // PAGE
    page1 = safe1 // PAGE
    offset0 = safe0 % PAGE
    offset1 = safe1 % PAGE
    request_valid = (req >= 0) & (req < num_requests)
    valid_group0 = request_valid & (token0 >= 0) & (page0 < TABLE_WIDTH)
    valid_group1 = tile1_active & request_valid & (token1 >= 0) & (page1 < TABLE_WIDTH)
    physical0 = tl.load(
        table_ptr + safe_req * str_ + tl.minimum(page0, TABLE_WIDTH - 1),
        mask=valid_group0,
        other=-1,
    )
    physical1 = tl.load(
        table_ptr + safe_req * str_ + tl.minimum(page1, TABLE_WIDTH - 1),
        mask=valid_group1,
        other=-1,
    )
    valid_group0 &= (physical0 >= 0) & (physical0 < cache_blocks)
    valid_group1 &= (physical1 >= 0) & (physical1 < cache_blocks)
    valid0_2d = valid_group0[:, None] & (within_group[None, :] < 4)
    valid1_2d = valid_group1[:, None] & (within_group[None, :] < 4)
    valid0 = tl.reshape(valid0_2d, (16,))
    valid1 = tl.reshape(valid1_2d, (16,))
    base0 = tl.maximum(physical0, 0).to(tl.int64)
    base1 = tl.maximum(physical1, 0).to(tl.int64)
    scale: tl.constexpr = (DIM**-0.5) * 1.4426950408889634

    keys0 = tl.load(
        k_ptr
        + (base0 * skb)[:, None, None]
        + (offset0[:, None, None] + within_group[None, :, None]) * skt
        + kv_head * skh
        + dim[None, None, :],
        mask=valid0_2d[:, :, None],
        other=0.0,
    )
    scores0 = tl.trans(tl.dot(tl.reshape(keys0, (16, DIM)), tl.trans(query))) * scale
    scores0 = tl.where(valid0[None, :], scores0, -1.0e20)
    keys1 = tl.load(
        k_ptr
        + (base1 * skb)[:, None, None]
        + (offset1[:, None, None] + within_group[None, :, None]) * skt
        + kv_head * skh
        + dim[None, None, :],
        mask=valid1_2d[:, :, None],
        other=0.0,
    )
    scores1 = tl.trans(tl.dot(tl.reshape(keys1, (16, DIM)), tl.trans(query))) * scale
    scores1 = tl.where(valid1[None, :], scores1, -1.0e20)
    next_max = tl.maximum(
        maximum,
        tl.maximum(tl.max(scores0, axis=1), tl.max(scores1, axis=1)),
    )
    alpha = tl.math.exp2(maximum - next_max)
    probabilities0 = tl.where(
        valid0[None, :],
        tl.math.exp2(scores0 - next_max[:, None]),
        0.0,
    )
    probabilities1 = tl.where(
        valid1[None, :],
        tl.math.exp2(scores1 - next_max[:, None]),
        0.0,
    )

    values0 = tl.load(
        v_ptr
        + (base0 * svb)[:, None, None]
        + (offset0[:, None, None] + within_group[None, :, None]) * svt
        + kv_head * svh
        + dim[None, None, :],
        mask=valid0_2d[:, :, None],
        other=0.0,
    )
    values0 = tl.reshape(values0, (16, DIM))
    accumulator = tl.trans(
        tl.dot(
            tl.trans(values0),
            tl.trans(probabilities0.to(values0.dtype)),
            acc=tl.trans(accumulator * alpha[:, None]),
        )
    )
    values1 = tl.load(
        v_ptr
        + (base1 * svb)[:, None, None]
        + (offset1[:, None, None] + within_group[None, :, None]) * svt
        + kv_head * svh
        + dim[None, None, :],
        mask=valid1_2d[:, :, None],
        other=0.0,
    )
    values1 = tl.reshape(values1, (16, DIM))
    accumulator = tl.trans(
        tl.dot(
            tl.trans(values1),
            tl.trans(probabilities1.to(values1.dtype)),
            acc=tl.trans(accumulator),
        )
    )
    denom = (
        denom * alpha + tl.sum(probabilities0, axis=1) + tl.sum(probabilities1, axis=1)
    )
    return accumulator, denom, next_max


@triton.jit(do_not_specialize=["num_rows", "num_requests"])
def _qsa_sparse_paged_mma_reverse_prefill_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    indices_ptr,
    table_ptr,
    req_ptr,
    partial_ptr,
    lse_ptr,
    out_ptr,
    sqr,
    sqh,
    skb,
    skt,
    skh,
    svb,
    svt,
    svh,
    sir,
    str_,
    sor,
    soh,
    num_rows,
    cache_blocks,
    num_requests,
    TOPK: tl.constexpr,
    PAGE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    DIM: tl.constexpr,
    TILES: tl.constexpr,
    SPLITS: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    GQA_GROUP_SIZE: tl.constexpr,
    HEAD_GROUPS: tl.constexpr,
    PAIR_TILES: tl.constexpr,
):
    row = tl.program_id(0)
    head_stream = tl.program_id(1)
    split = tl.program_id(2)
    kv_head = head_stream // HEAD_GROUPS
    head_group = head_stream % HEAD_GROUPS
    head0 = kv_head * GQA_GROUP_SIZE + head_group * 16
    req = tl.load(req_ptr + row)
    safe_req = tl.minimum(tl.maximum(req, 0), num_requests - 1)
    count = tl.minimum(TOPK, tl.maximum(0, tl.load(indices_ptr + row * sir + TOPK)))
    dim = tl.arange(0, DIM)
    heads = tl.arange(0, 16)
    query = tl.load(
        q_ptr + row * sqr + (head0 + heads[:, None]) * sqh + dim[None, :],
        mask=(head_group * 16 + heads[:, None]) < GQA_GROUP_SIZE,
        other=0.0,
    )
    maximum = tl.full((16,), -1.0e20, tl.float32)
    denom = tl.zeros((16,), tl.float32)
    accumulator = tl.zeros((16, DIM), tl.float32)

    tile_end = tl.minimum(TILES, tl.cdiv(count, 16))
    full_tiles = tl.minimum(TILES, count // 16)
    # Split 0 owns the only possibly partial tile. The remaining tiles are
    # complete and traverse from high to low index in round-robin split order.
    if (split == 0) & (tile_end > full_tiles):
        accumulator, denom, maximum = _qsa_sparse_paged_mma_reverse_tile(
            q_ptr,
            k_ptr,
            v_ptr,
            indices_ptr,
            table_ptr,
            query,
            accumulator,
            denom,
            maximum,
            row,
            kv_head,
            req,
            safe_req,
            count,
            full_tiles,
            skb,
            skt,
            skh,
            svb,
            svt,
            svh,
            sir,
            str_,
            cache_blocks,
            num_requests,
            PAGE=PAGE,
            TABLE_WIDTH=TABLE_WIDTH,
            DIM=DIM,
            MASK_COUNT=True,
            TRANSPOSE_QK=True,
        )
    if PAIR_TILES:
        for reverse_rank in range(split, full_tiles, 2 * SPLITS):
            tile0 = full_tiles - 1 - reverse_rank
            tile1 = tile0 - SPLITS
            accumulator, denom, maximum = _qsa_sparse_paged_mma_reverse_two_tiles(
                k_ptr,
                v_ptr,
                indices_ptr,
                table_ptr,
                query,
                accumulator,
                denom,
                maximum,
                row,
                kv_head,
                req,
                safe_req,
                tile0,
                tile1,
                reverse_rank + SPLITS < full_tiles,
                skb,
                skt,
                skh,
                svb,
                svt,
                svh,
                sir,
                str_,
                cache_blocks,
                num_requests,
                PAGE=PAGE,
                TABLE_WIDTH=TABLE_WIDTH,
                DIM=DIM,
            )
    else:
        for reverse_rank in range(split, full_tiles, SPLITS):
            tile = full_tiles - 1 - reverse_rank
            accumulator, denom, maximum = _qsa_sparse_paged_mma_reverse_tile(
                q_ptr,
                k_ptr,
                v_ptr,
                indices_ptr,
                table_ptr,
                query,
                accumulator,
                denom,
                maximum,
                row,
                kv_head,
                req,
                safe_req,
                count,
                tile,
                skb,
                skt,
                skh,
                svb,
                svt,
                svh,
                sir,
                str_,
                cache_blocks,
                num_requests,
                PAGE=PAGE,
                TABLE_WIDTH=TABLE_WIDTH,
                DIM=DIM,
                MASK_COUNT=False,
                TRANSPOSE_QK=True,
            )

    has_values = denom > 0
    result = tl.where(
        has_values[:, None],
        accumulator / tl.maximum(denom[:, None], 1.0e-20),
        0.0,
    )
    active_heads = head_group * 16 + heads < GQA_GROUP_SIZE
    if SPLITS == 1:
        tl.store(
            out_ptr + row * sor + (head0 + heads[:, None]) * soh + dim[None, :],
            result,
            mask=active_heads[:, None],
        )
    else:
        partial_row = (split * num_rows + row).to(tl.int64)
        tl.store(
            partial_ptr
            + (partial_row * NUM_QUERY_HEADS + head0 + heads[:, None]) * DIM
            + dim[None, :],
            result,
            mask=active_heads[:, None],
        )
        lse = tl.where(
            has_values,
            maximum + tl.math.log2(tl.maximum(denom, 1.0e-20)),
            -float("inf"),
        )
        tl.store(
            lse_ptr + partial_row * NUM_QUERY_HEADS + head0 + heads,
            lse,
            mask=active_heads,
        )


@triton.jit(do_not_specialize=["num_rows"])
def _qsa_merge_splitk_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    stride_output_row,
    stride_output_head,
    num_rows,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    head = tl.program_id(1)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    dim_offsets = tl.arange(0, HEAD_DIM)
    split_mask = split_offsets < NUM_SPLITS
    lse = tl.load(
        partial_lse_ptr + (split_offsets * num_rows + row) * NUM_QUERY_HEADS + head,
        mask=split_mask,
        other=-float("inf"),
    )
    lse_max = tl.max(lse, axis=0)
    has_values = lse_max > -float("inf")
    shifted = tl.where(split_mask & has_values, lse - lse_max, -float("inf"))
    weights = tl.math.exp2(shifted)
    denominator = tl.sum(weights, axis=0)
    split_rows = split_offsets.to(tl.int64) * num_rows + row
    partial_output = tl.load(
        partial_output_ptr
        + (split_rows[:, None] * NUM_QUERY_HEADS + head) * HEAD_DIM
        + dim_offsets[None, :],
        mask=split_mask[:, None],
        other=0.0,
    )
    merged = tl.sum(partial_output * weights[:, None], axis=0)
    merged = tl.where(denominator > 0, merged / denominator, 0.0)
    tl.store(
        output_ptr + row * stride_output_row + head * stride_output_head + dim_offsets,
        merged,
    )


@triton.jit(do_not_specialize=["num_rows", "num_requests"])
def _qsa_build_flash_subpage_table_kernel(
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    subpage_table_ptr,
    context_lens_ptr,
    stride_indices_row,
    stride_table_req,
    stride_subpage_row,
    num_rows,
    num_cache_blocks,
    num_requests,
    SELECTION_WIDTH: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    NUM_SUBPAGES: tl.constexpr,
    BLOCK_RANKS: tl.constexpr,
) -> None:
    """Map expanded logical-token groups to physical cache subpages."""

    row = tl.program_id(0)
    ranks = tl.program_id(1) * BLOCK_RANKS + tl.arange(0, BLOCK_RANKS)
    row_valid = row < num_rows
    request = tl.load(token_to_req_ptr + row, mask=row_valid, other=-1)
    request_valid = row_valid & (request >= 0) & (request < num_requests)
    safe_request = tl.maximum(request, 0)

    valid_count = tl.load(
        indices_ptr + row * stride_indices_row + SELECTION_WIDTH,
        mask=row_valid,
        other=0,
    )
    valid_count = tl.maximum(0, tl.minimum(valid_count, SELECTION_WIDTH))
    if tl.program_id(1) == 0:
        tl.store(context_lens_ptr + row, valid_count, mask=row_valid)

    group_count = (valid_count + COMPRESS_RATIO - 1) // COMPRESS_RATIO
    active = ranks < group_count
    logical_token = tl.load(
        indices_ptr + row * stride_indices_row + ranks * COMPRESS_RATIO,
        mask=row_valid & active,
        other=-1,
    )
    logical_page = tl.maximum(logical_token, 0) // PAGE_SIZE
    page_valid = (
        request_valid
        & active
        & (logical_token >= 0)
        & (logical_token % COMPRESS_RATIO == 0)
        & (logical_page < PAGE_TABLE_WIDTH)
    )
    physical_page = tl.load(
        block_table_ptr
        + safe_request * stride_table_req
        + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
        mask=page_valid,
        other=-1,
    )
    physical_page_valid = (
        page_valid & (physical_page >= 0) & (physical_page < num_cache_blocks)
    )
    subpages_per_page: tl.constexpr = PAGE_SIZE // COMPRESS_RATIO
    physical_subpage = (
        physical_page * subpages_per_page
        + (tl.maximum(logical_token, 0) % PAGE_SIZE) // COMPRESS_RATIO
    )
    tl.store(
        subpage_table_ptr + row * stride_subpage_row + ranks,
        tl.where(physical_page_valid, physical_subpage, -1),
        mask=ranks < NUM_SUBPAGES,
    )


@triton.jit
def _store_qsa_rows_kernel(
    cache_ptr,
    slots_ptr,
    rows_ptr,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_rows_row,
    stride_rows_dim,
    num_rows,
    num_blocks,
    PAGE_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    slot = tl.load(slots_ptr + row)
    valid = (row < num_rows) & (slot >= 0) & (slot < num_blocks * PAGE_SIZE)
    block = tl.maximum(slot, 0) // PAGE_SIZE
    token = tl.maximum(slot, 0) % PAGE_SIZE
    values = tl.load(
        rows_ptr + row * stride_rows_row + dims * stride_rows_dim,
        mask=valid & (dims < WIDTH),
        other=0,
    )
    tl.store(
        cache_ptr
        + block * stride_cache_block
        + token * stride_cache_token
        + dims * stride_cache_dim,
        values,
        mask=valid & (dims < WIDTH),
    )


@triton.jit
def _compress_qsa_groups_kernel(
    raw_keys_ptr,  # this step's raw key rows, straight from activations
    raw_positions_ptr,  # this step's per-token positions
    compressor_state_cache_ptr,  # per-request ring of previous raw keys
    rope_cache_ptr,  # packed RoPE position tail of the ring
    compressor_state_table_ptr,
    token_to_req_ptr,
    query_start_loc_ptr,
    logical_positions_ptr,
    compressed_slots_ptr,
    pooled_ptr,
    first_positions_ptr,
    stride_raw_row,
    stride_raw_dim,
    stride_raw_positions_row,
    stride_raw_positions_dim,
    stride_compressor_state_block,
    stride_compressor_state_token,
    stride_compressor_state_dim,
    stride_rope_block,
    stride_rope_token,
    stride_rope_dim,
    stride_compressor_state_table_req,
    stride_pooled_row,
    stride_pooled_dim,
    stride_positions_row,
    stride_positions_dim,
    num_rows,
    num_compressor_state_blocks,
    num_requests,
    COMPRESSOR_STATE_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOAD_ROPE_POSITIONS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    request = tl.load(token_to_req_ptr + row)
    end_position = tl.load(logical_positions_ptr + row)
    compressed_slot = tl.load(compressed_slots_ptr + row)
    valid_request = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_row_start = tl.load(
        query_start_loc_ptr + safe_request, mask=valid_request, other=0
    )
    query_row_end = tl.load(
        query_start_loc_ptr + safe_request + 1, mask=valid_request, other=0
    )
    chunk_start_position = end_position - (row - query_row_start)
    compressor_state_block = tl.load(
        compressor_state_table_ptr + safe_request * stride_compressor_state_table_req,
        mask=valid_request,
        other=-1,
    )
    valid_compressor_state_block = (compressor_state_block >= 0) & (
        compressor_state_block < num_compressor_state_blocks
    )
    valid_row = (
        (row < num_rows)
        & valid_request
        & (row >= query_row_start)
        & (row < query_row_end)
        & (end_position >= COMPRESS_RATIO - 1)
        & (compressed_slot >= 0)
    )
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # A group can span the compressor-state ring (older members) and this
    # step's raw rows (members at positions >= chunk_start_position).
    for group_offset in tl.range(0, COMPRESS_RATIO):
        position = end_position - (COMPRESS_RATIO - 1 - group_offset)
        use_raw = position >= chunk_start_position
        raw_row = query_row_start + position - chunk_start_position
        raw_values = tl.load(
            raw_keys_ptr + raw_row * stride_raw_row + dims * stride_raw_dim,
            mask=valid_row
            & use_raw
            & (raw_row >= query_row_start)
            & (raw_row < query_row_end)
            & (raw_row < num_rows)
            & (dims < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        compressor_state_values = tl.load(
            compressor_state_cache_ptr
            + tl.maximum(compressor_state_block, 0).to(tl.int64)
            * stride_compressor_state_block
            + (position % COMPRESSOR_STATE_SIZE) * stride_compressor_state_token
            + dims * stride_compressor_state_dim,
            mask=valid_row
            & ~use_raw
            & valid_compressor_state_block
            & (dims < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.where(use_raw, raw_values, compressor_state_values)

    tl.store(
        pooled_ptr + row * stride_pooled_row + dims * stride_pooled_dim,
        accumulator / COMPRESS_RATIO,
        mask=(row < num_rows) & (dims < HEAD_DIM),
    )

    position_dims = tl.arange(0, 4)
    first_position = end_position - COMPRESS_RATIO + 1
    if LOAD_ROPE_POSITIONS:
        first_from_raw = first_position >= chunk_start_position
        raw_first_row = query_row_start + first_position - chunk_start_position
        raw_position_values = tl.load(
            raw_positions_ptr
            + raw_first_row * stride_raw_positions_row
            + position_dims * stride_raw_positions_dim,
            mask=valid_row
            & first_from_raw
            & (raw_first_row >= query_row_start)
            & (raw_first_row < query_row_end)
            & (raw_first_row < num_rows)
            & (position_dims < 3),
            other=0,
        )
        compressor_state_position_values = tl.load(
            rope_cache_ptr
            + tl.maximum(compressor_state_block, 0).to(tl.int64) * stride_rope_block
            + (first_position % COMPRESSOR_STATE_SIZE) * stride_rope_token
            + position_dims * stride_rope_dim,
            mask=valid_row
            & ~first_from_raw
            & valid_compressor_state_block
            & (position_dims < 3),
            other=0,
        )
        position_values = tl.where(
            first_from_raw,
            raw_position_values,
            compressor_state_position_values,
        )
    else:
        position_values = tl.where(valid_row, first_position, 0)
    tl.store(
        first_positions_ptr
        + row * stride_positions_row
        + position_dims * stride_positions_dim,
        position_values,
        mask=(row < num_rows) & (position_dims < 3),
    )


def _select_attention_backend(
    num_rows: int,
    num_kv_heads: int,
    group_size: int,
    selection_count_hint: int | None,
) -> str:
    """Choose FMA/MMA from host workload metadata, independent of phase.

    C500 sweeps compare the complete backend calls, including split/merge.
    Small head groups need longer selections to amortize the padded MMA tile;
    larger groups benefit earlier from sharing K/V across query heads. An
    unknown length keeps the universal FMA path; capacity is not a live count.
    """
    if group_size <= 1 or selection_count_hint is None:
        return "fma"
    count = max(selection_count_hint, 0)
    streams = num_rows * num_kv_heads * triton.cdiv(group_size, 16)
    if group_size <= 3:
        # Group2 needs row/stream parallelism even for 1024-token selections.
        # Group3 shares its FMA payload across three heads and retains the
        # established long-selection crossover against that fused baseline.
        min_streams = 256 if group_size == 2 else 1
        use_mma = (
            count >= 1024 and streams >= min_streams
        ) or streams * count >= 400_000
    elif group_size == 4:
        use_mma = (streams >= 32 and count >= 1024) or (streams >= 256 and count >= 256)
    else:
        use_mma = (
            count >= 2048
            or (streams >= 32 and count >= 256)
            or (streams >= 96 and count >= 64)
        )
    return "mma" if use_mma else "fma"


def _select_config(
    num_rows: int,
    num_kv_heads: int,
    use_prefill_config: bool,
    num_columns: int,
    *,
    backend: str = "fma",
    selection_count_hint: int | None = None,
) -> tuple[int, int, int, int]:
    """Choose the C500 split-K shape without changing reduction coverage.

    ``use_prefill_config`` separates long-query batches from decode/verify
    when their measured dependency-chain tradeoff differs.  A hint only caps
    split launches; the device-side packed count still bounds every load and
    ``num_tiles`` stays capacity-based, so a low hint cannot truncate
    attention.
    """
    if backend not in ("fma", "mma"):
        raise ValueError(f"unknown QSA backend: {backend}")
    programs = num_rows * num_kv_heads
    small_per_head = backend == "fma" and programs <= 4
    block_n, warps = (64, 4) if small_per_head else (16, 1)
    target_splits = (
        16
        if programs <= 128
        else 8
        if programs <= 512
        else 4
        if programs <= 1024
        else 2
        if programs <= 2048
        else 1
    )
    num_tiles = triton.cdiv(num_columns, block_n)
    if selection_count_hint is not None:
        useful = min(max(selection_count_hint, 0), num_columns)
        # Workload-based C500 scheduling shared by every FMA group size. No
        # selection entries are dropped: this only changes the round-robin
        # split count, while NUM_TILES remains based on the packed capacity.
        if (
            backend == "fma"
            and use_prefill_config
            and 1024 < programs <= 4096
            and useful >= 1536
        ):
            # Long prefill has ample row parallelism, but one online-softmax
            # chain per row still leaves instruction-level parallelism on the
            # table.  Four KV partitions shorten that chain enough to cover
            # the partial/LSE write and merge cost on C500.  Above this band
            # the extra workspace and merge traffic dominate the small gain.
            target_splits = 4
        elif backend == "fma" and 80 <= programs <= 104 and useful <= 512:
            target_splits = 8
        elif backend == "fma" and 108 <= programs <= 116 and useful >= 1536:
            target_splits = 14
        elif backend == "fma" and 116 < programs <= 120 and useful >= 1536:
            target_splits = 24
        target_splits = min(target_splits, max(1, triton.cdiv(useful, block_n)))
    return block_n, warps, num_tiles, min(target_splits, num_tiles)


def _merge_warps(num_splits: int) -> int:
    """Keep small reductions within one C500 wavefront."""
    return 1 if num_splits <= 8 else 2


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "1").lower() not in ("0", "false")


def _qsa_flash_subpage_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    compress_ratio: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Run exact QSA attention through FlashAttention on physical subpages.

    QSA expands every selected compressed group into ``compress_ratio``
    consecutive logical tokens. Reinterpreting a cache page as smaller physical
    subpages therefore lets FlashAttention consume one block-table entry per
    selected group, while ``cache_seqlens`` masks the final causal-tail group.
    """

    from flash_attn import flash_attn_with_kvcache

    selection_width = logical_indices.shape[1] - 1
    num_subpages = (selection_width + compress_ratio - 1) // compress_ratio
    subpage_table = torch.empty(
        (q.shape[0], num_subpages), dtype=torch.int32, device=q.device
    )
    context_lens = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
    block_ranks = 256
    _qsa_build_flash_subpage_table_kernel[
        (q.shape[0], triton.cdiv(num_subpages, block_ranks))
    ](
        logical_indices,
        block_table,
        token_to_req,
        subpage_table,
        context_lens,
        logical_indices.stride(0),
        block_table.stride(0),
        subpage_table.stride(0),
        q.shape[0],
        k_cache.shape[0],
        block_table.shape[0],
        SELECTION_WIDTH=selection_width,
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        COMPRESS_RATIO=compress_ratio,
        NUM_SUBPAGES=num_subpages,
        BLOCK_RANKS=block_ranks,
        num_warps=4,
        num_stages=1,
    )

    subpages_per_page = k_cache.shape[1] // compress_ratio
    subpage_shape = (
        k_cache.shape[0] * subpages_per_page,
        compress_ratio,
        k_cache.shape[2],
        k_cache.shape[3],
    )
    subpage_strides = (
        compress_ratio * k_cache.stride(1),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
    )
    key_subpages = torch.as_strided(k_cache, size=subpage_shape, stride=subpage_strides)
    value_subpages = torch.as_strided(
        v_cache, size=subpage_shape, stride=subpage_strides
    )
    flash_output = flash_attn_with_kvcache(
        q=q[:, None],
        k_cache=key_subpages,
        v_cache=value_subpages,
        block_table=subpage_table,
        cache_seqlens=context_lens,
        softmax_scale=q.shape[-1] ** -0.5,
        causal=False,
        num_splits=0,
    )
    out.copy_(flash_output[:, 0])
    return out


def qsa_sparse_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    use_prefill_config: bool,
    out: torch.Tensor | None = None,
    compress_ratio: int | None = None,
    selection_count_hint: int | None = None,
) -> torch.Tensor:
    """Run sparse GQA directly over paged BF16 K/V caches.

    logical_indices is the PACKED selection buffer: [rows, selection_width + 1]
    with the trailing column holding each row's valid-entry count (written by
    the expand kernel; never a token index). The FMA backend uses FP32 vector
    reductions; the MMA backend shares each K/V tile across query heads.
    use_prefill_config is retained for upstream call compatibility; this MetaX
    schedule depends on row count and selection width. ``selection_count_hint``
    is optional host metadata used only to avoid idle split programs.
    """
    # ==== [QSA-DBG 20261009] debug block: print shapes/non-tensor args + dump prefill pt (comment out after debug) ====
    # _dbg_rank = os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "0"
    # print(
    #     f"[QSA-DBG r{_dbg_rank}] qsa_sparse_paged_attention: "
    #     f"inputs q{tuple(q.shape)}/{q.dtype} "
    #     f"k_cache{tuple(k_cache.shape)}/{k_cache.dtype} "
    #     f"v_cache{tuple(v_cache.shape)}/{v_cache.dtype} "
    #     f"logical_indices{tuple(logical_indices.shape)}/{logical_indices.dtype} "
    #     f"block_table{tuple(block_table.shape)}/{block_table.dtype} "
    #     f"token_to_req{tuple(token_to_req.shape)}/{token_to_req.dtype}; "
    #     f"output out{tuple(out.shape) if out is not None else tuple(q.shape)}; "
    #     f"non-tensor args: use_prefill_config={use_prefill_config}, "
    #     f"compress_ratio={compress_ratio}, "
    #     f"selection_count_hint={selection_count_hint}",
    #     flush=True,
    # )
    # if use_prefill_config:
    #     # prefill never runs under cudagraph capture; the guard below is a
    #     # belt-and-suspenders check so torch.save can never hit a capture.
    #     try:
    #         _dbg_capturing = torch.cuda.is_current_stream_capturing()
    #     except Exception:
    #         _dbg_capturing = False
    #     try:
    #         if not _dbg_capturing:
    #             _dbg_dir = "/workspace/qsa_dbg_pt"
    #             os.makedirs(_dbg_dir, exist_ok=True)
    #             _dbg_n = getattr(qsa_sparse_paged_attention, "_dbg_n", 0)
    #             _dbg_max = int(os.environ.get("VLLM_QSA_DBG_MAX_SAVES", "30"))
    #             if _dbg_n < _dbg_max:
    #                 qsa_sparse_paged_attention._dbg_n = _dbg_n + 1
    #                 _dbg_path = os.path.join(
    #                     _dbg_dir,
    #                     f"prefill_r{_dbg_rank}_{_dbg_n:02d}_T{q.shape[0]}.pt",
    #                 )
    #                 torch.save(
    #                     {
    #                         "logical_indices": logical_indices.detach().cpu().clone(),
    #                         "token_to_req": token_to_req.detach().cpu().clone(),
    #                         "meta": {
    #                             "rank": _dbg_rank,
    #                             "save_index": _dbg_n,
    #                             "q_shape": tuple(q.shape),
    #                             "k_cache_shape": tuple(k_cache.shape),
    #                             "v_cache_shape": tuple(v_cache.shape),
    #                             "logical_indices_shape": tuple(logical_indices.shape),
    #                             "block_table_shape": tuple(block_table.shape),
    #                             "token_to_req_shape": tuple(token_to_req.shape),
    #                             "use_prefill_config": use_prefill_config,
    #                             "compress_ratio": compress_ratio,
    #                             "selection_count_hint": selection_count_hint,
    #                             "note": (
    #                                 "logical_indices: packed selection "
    #                                 "[T, W+1] int32, trailing col = valid "
    #                                 "count; token_to_req: per-token request "
    #                                 "id int32; saved on prefill calls only"
    #                             ),
    #                         },
    #                     },
    #                     _dbg_path,
    #                 )
    #                 print(
    #                     f"[QSA-DBG r{_dbg_rank}] torch.save -> {_dbg_path}",
    #                     flush=True,
    #                 )
    #     except Exception as _dbg_e:
    #         print(
    #             f"[QSA-DBG r{_dbg_rank}] pt save FAILED: "
    #             f"{type(_dbg_e).__name__}: {_dbg_e}",
    #             flush=True,
    #         )
    # ==== [QSA-DBG 20261009] end debug block ====
    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("QSA sparse attention received invalid Q/K/V shapes")
    if logical_indices.ndim != 2 or logical_indices.shape[0] != q.shape[0]:
        raise ValueError("QSA indices must have one row per query")
    if token_to_req.shape != (q.shape[0],) or block_table.ndim != 2:
        raise ValueError("QSA sparse attention metadata has invalid shapes")
    if not all(k_cache.shape[:3]) or not all(block_table.shape):
        raise ValueError("QSA sparse attention cache and block table must be nonempty")
    if logical_indices.shape[1] < 2:
        raise ValueError(
            "QSA packed indices need selection columns plus the count column"
        )
    if q.shape[2] != k_cache.shape[3] or q.shape[1] % k_cache.shape[2]:
        raise ValueError("QSA sparse attention requires valid grouped-query heads")
    head_dim = q.shape[2]
    assert head_dim >= 16 and (head_dim & (head_dim - 1)) == 0
    assert q.dtype == k_cache.dtype == v_cache.dtype == torch.bfloat16
    assert logical_indices.dtype == block_table.dtype == torch.int32
    assert token_to_req.dtype == torch.int32
    assert q.device == k_cache.device == v_cache.device
    assert q.device == logical_indices.device == block_table.device
    assert q.device == token_to_req.device
    assert q.stride(2) == k_cache.stride(3) == v_cache.stride(3) == 1
    assert logical_indices.stride(1) == block_table.stride(1) == 1
    assert token_to_req.stride(0) == 1

    if out is None:
        out = torch.empty_like(q)
    if out.shape != q.shape:
        raise ValueError("QSA sparse output must match its query")
    assert out.dtype == q.dtype and out.device == q.device
    assert out.stride(2) == 1
    if not q.shape[0]:
        return out

    use_flash_subpages = (
        compress_ratio == 4
        and k_cache.shape[1] % compress_ratio == 0
        and k_cache.stride() == v_cache.stride()
        and k_cache.stride(0) == k_cache.shape[1] * k_cache.stride(1)
        and _env_enabled("VLLM_METAX_QSA_FLASH_SUBPAGES")
    )
    if use_flash_subpages:
        return _qsa_flash_subpage_attention(
            q,
            k_cache,
            v_cache,
            logical_indices,
            block_table,
            token_to_req,
            compress_ratio,
            out,
        )

    group_size = q.shape[1] // k_cache.shape[2]
    selection_width = logical_indices.shape[1] - 1  # trailing column is the count
    useful_selection_count = (
        None
        if selection_count_hint is None
        else min(max(selection_count_hint, 0), selection_width)
    )
    # Production has two backends. FMA is the universal vector path; MMA
    # pads grouped query heads to a native sixteen-row tile when the supported
    # grouped-token layout has enough work to amortize it.
    fusion_enabled = _env_enabled("VLLM_METAX_QSA_FUSED_GQA")
    optimized_layout = (
        fusion_enabled
        and compress_ratio == 4
        and q.shape[2] == 256
        and k_cache.shape[1] % 4 == 0
    )
    enough_work_for_mma = useful_selection_count is not None and (
        useful_selection_count >= 1024 or q.shape[0] * useful_selection_count >= 400_000
    )
    use_mma = optimized_layout and (
        _select_attention_backend(
            q.shape[0], k_cache.shape[2], group_size, useful_selection_count
        )
        == "mma"
    )
    backend = "mma" if use_mma else "fma"
    heads_per_stream = 16 if use_mma else 3 if group_size == 3 else 1
    head_groups = triton.cdiv(group_size, heads_per_stream)
    kernel_streams = k_cache.shape[2] * head_groups
    block_n, partial_warps, num_tiles, num_splits = _select_config(
        q.shape[0],
        kernel_streams,
        use_prefill_config,
        selection_width,
        backend=backend,
        selection_count_hint=selection_count_hint,
    )
    # Split=1 writes output directly and compiles out all workspace accesses.
    if num_splits == 1:
        partial_output = out
        partial_lse = out
    else:
        # FP32 partials preserve accuracy when merging independently normalized
        # splits.
        partial_output = torch.empty(
            (num_splits, *q.shape), dtype=torch.float32, device=q.device
        )
        partial_lse = torch.empty(
            (num_splits, q.shape[0], q.shape[1]),
            dtype=torch.float32,
            device=q.device,
        )

    common_args = (
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        partial_output,
        partial_lse,
        out,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        logical_indices.stride(0),
        block_table.stride(0),
        out.stride(0),
        out.stride(1),
        q.shape[0],
        k_cache.shape[0],
        block_table.shape[0],
    )
    if use_mma:
        # Every MMA shape uses the same reverse implementation. Pairing is a
        # workload decision, independent of the GQA group size.
        _qsa_sparse_paged_mma_reverse_prefill_kernel[
            (q.shape[0], kernel_streams, num_splits)
        ](
            q,
            k_cache,
            v_cache,
            logical_indices,
            block_table,
            token_to_req,
            partial_output,
            partial_lse,
            out,
            q.stride(0),
            q.stride(1),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            logical_indices.stride(0),
            block_table.stride(0),
            out.stride(0),
            out.stride(1),
            q.shape[0],
            k_cache.shape[0],
            block_table.shape[0],
            TOPK=selection_width,
            PAGE=k_cache.shape[1],
            TABLE_WIDTH=block_table.shape[1],
            DIM=q.shape[2],
            TILES=num_tiles,
            SPLITS=num_splits,
            NUM_QUERY_HEADS=q.shape[1],
            GQA_GROUP_SIZE=group_size,
            HEAD_GROUPS=head_groups,
            PAIR_TILES=enough_work_for_mma,
            num_warps=2,
            num_stages=1,
        )
    else:
        shared_kv_page_stride = k_cache.stride(0) == v_cache.stride(0)
        _qsa_sparse_paged_fma_kernel[(q.shape[0], kernel_streams, num_splits)](
            *common_args,
            TOPK=selection_width,
            PAGE=k_cache.shape[1],
            TABLE_WIDTH=block_table.shape[1],
            DIM=q.shape[2],
            SPLITS=num_splits,
            TILES=num_tiles,
            BN=block_n,
            GROUPS=block_n // 4,
            NUM_QUERY_HEADS=q.shape[1],
            GQA_GROUP_SIZE=group_size,
            HEAD_GROUPS=head_groups,
            HEADS_PER_STREAM=heads_per_stream,
            SHARED_KV_PAGE_STRIDE=shared_kv_page_stride,
            GROUPED_TOKENS=optimized_layout,
            num_warps=partial_warps,
            num_stages=1,
        )
    if num_splits == 1:
        return out

    _qsa_merge_splitk_kernel[(q.shape[0], q.shape[1])](
        partial_output,
        partial_lse,
        out,
        out.stride(0),
        out.stride(1),
        q.shape[0],
        HEAD_DIM=q.shape[2],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        BLOCK_SPLITS=triton.next_power_of_2(num_splits),
        num_warps=_merge_warps(num_splits),
        num_stages=1,
    )
    return out


def warmup_qsa_sparse_paged_attention(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    *,
    num_query_heads: int,
    selection_width: int,
) -> tuple[tuple[int, int, int], ...]:
    """Compile every production-reachable FMA, MMA, and merge shape."""

    head_dim = kv_cache.shape[-1] // 2
    key_cache, value_cache = kv_cache.transpose(1, 2).split(head_dim, dim=-1)
    num_kv_heads = key_cache.shape[2]
    group_size = num_query_heads // num_kv_heads
    count_hints = {
        None,
        0,
        # A live-count hint can cap a split launch at any count from 1 to 16.
        # Cover intermediate caps such as 5/6/7, for both FMA tile widths.
        *(
            min(selection_width, block_n * splits)
            for block_n in (16, 64)
            for splits in range(1, 17)
        ),
        min(selection_width, 1536),
        selection_width,
    }

    num_rows = 16
    num_requests = 16
    q_ptr = TritonWarmupTensor(
        torch.bfloat16, shape=(num_rows, num_query_heads, head_dim)
    )
    k_cache_ptr = TritonWarmupTensor(key_cache.dtype, shape=tuple(key_cache.shape))
    v_cache_ptr = TritonWarmupTensor(value_cache.dtype, shape=tuple(value_cache.shape))
    indices_ptr = TritonWarmupTensor(torch.int32, shape=(num_rows, selection_width + 1))
    block_table_ptr = TritonWarmupTensor(
        block_table.dtype, shape=tuple(block_table.shape)
    )
    token_to_req_ptr = TritonWarmupTensor(torch.int32)
    output_ptr = TritonWarmupTensor(
        torch.bfloat16, shape=(num_rows, num_query_heads, head_dim)
    )
    head_stride = head_dim
    row_stride = num_query_heads * head_dim
    num_cache_blocks = triton_scalar_specialization_rep(kv_cache.shape[0])
    shared_kv_page_stride = key_cache.stride(0) == value_cache.stride(0)
    warmed: set[tuple[int, int, int]] = set()
    merge_profiles: dict[int, tuple[TritonWarmupTensor, TritonWarmupTensor]] = {}

    def workspace(num_splits: int):
        if num_splits == 1:
            return output_ptr, output_ptr
        partials = merge_profiles.get(num_splits)
        if partials is None:
            partials = (
                TritonWarmupTensor(
                    torch.float32,
                    shape=(num_splits, num_rows, num_query_heads, head_dim),
                ),
                TritonWarmupTensor(
                    torch.float32,
                    shape=(num_splits, num_rows, num_query_heads),
                ),
            )
            merge_profiles[num_splits] = partials
        return partials

    heads_per_fma_stream = 3 if group_size == 3 else 1
    fma_head_groups = triton.cdiv(group_size, heads_per_fma_stream)
    fma_streams = num_kv_heads * fma_head_groups
    fma_profiles = {
        _select_config(
            rows,
            fma_streams,
            use_prefill,
            selection_width,
            backend="fma",
            selection_count_hint=count_hint,
        )
        for rows in range(1, 8193)
        for use_prefill in (False, True)
        for count_hint in count_hints
    }
    grouped_token_options = (
        (False, True) if head_dim == 256 and key_cache.shape[1] % 4 == 0 else (False,)
    )
    for block_n, warps, num_tiles, num_splits in sorted(fma_profiles):
        partial_output_ptr, partial_lse_ptr = workspace(num_splits)
        for grouped_tokens in grouped_token_options:
            _qsa_sparse_paged_fma_kernel.warmup(
                q_ptr,
                k_cache_ptr,
                v_cache_ptr,
                indices_ptr,
                block_table_ptr,
                token_to_req_ptr,
                partial_output_ptr,
                partial_lse_ptr,
                output_ptr,
                row_stride,
                head_stride,
                key_cache.stride(0),
                key_cache.stride(1),
                key_cache.stride(2),
                value_cache.stride(0),
                value_cache.stride(1),
                value_cache.stride(2),
                selection_width + 1,
                block_table.stride(0),
                row_stride,
                head_stride,
                num_rows,
                num_cache_blocks,
                num_requests,
                TOPK=selection_width,
                PAGE=key_cache.shape[1],
                TABLE_WIDTH=block_table.shape[1],
                DIM=head_dim,
                SPLITS=num_splits,
                TILES=num_tiles,
                BN=block_n,
                GROUPS=block_n // 4,
                NUM_QUERY_HEADS=num_query_heads,
                GQA_GROUP_SIZE=group_size,
                HEAD_GROUPS=fma_head_groups,
                HEADS_PER_STREAM=heads_per_fma_stream,
                SHARED_KV_PAGE_STRIDE=shared_kv_page_stride,
                GROUPED_TOKENS=grouped_tokens,
                num_warps=warps,
                num_stages=1,
                grid=(num_rows, fma_streams, num_splits),
            )
        warmed.add((block_n, num_splits, warps))

    mma_compatible = group_size >= 2 and head_dim == 256 and key_cache.shape[1] % 4 == 0
    if mma_compatible:
        mma_head_groups = triton.cdiv(group_size, 16)
        mma_streams = num_kv_heads * mma_head_groups
        mma_profiles = {
            _select_config(
                rows,
                mma_streams,
                True,
                selection_width,
                backend="mma",
                selection_count_hint=count_hint,
            )
            for rows in range(1, 8193)
            for count_hint in count_hints
        }
        for _block_n, _warps, num_tiles, num_splits in sorted(mma_profiles):
            partial_output_ptr, partial_lse_ptr = workspace(num_splits)
            for pair_tiles in (False, True):
                _qsa_sparse_paged_mma_reverse_prefill_kernel.warmup(
                    q_ptr,
                    k_cache_ptr,
                    v_cache_ptr,
                    indices_ptr,
                    block_table_ptr,
                    token_to_req_ptr,
                    partial_output_ptr,
                    partial_lse_ptr,
                    output_ptr,
                    row_stride,
                    head_stride,
                    key_cache.stride(0),
                    key_cache.stride(1),
                    key_cache.stride(2),
                    value_cache.stride(0),
                    value_cache.stride(1),
                    value_cache.stride(2),
                    selection_width + 1,
                    block_table.stride(0),
                    row_stride,
                    head_stride,
                    num_rows,
                    num_cache_blocks,
                    num_requests,
                    TOPK=selection_width,
                    PAGE=key_cache.shape[1],
                    TABLE_WIDTH=block_table.shape[1],
                    DIM=head_dim,
                    TILES=num_tiles,
                    SPLITS=num_splits,
                    NUM_QUERY_HEADS=num_query_heads,
                    GQA_GROUP_SIZE=group_size,
                    HEAD_GROUPS=mma_head_groups,
                    PAIR_TILES=pair_tiles,
                    num_warps=2,
                    num_stages=1,
                    grid=(num_rows, mma_streams, num_splits),
                )
            warmed.add((16, num_splits, 2))

    for num_splits, (partial_output_ptr, partial_lse_ptr) in merge_profiles.items():
        _qsa_merge_splitk_kernel.warmup(
            partial_output_ptr,
            partial_lse_ptr,
            output_ptr,
            row_stride,
            head_stride,
            num_rows,
            HEAD_DIM=head_dim,
            NUM_QUERY_HEADS=num_query_heads,
            NUM_SPLITS=num_splits,
            BLOCK_SPLITS=triton.next_power_of_2(num_splits),
            num_warps=_merge_warps(num_splits),
            num_stages=1,
            grid=(num_rows, num_query_heads),
        )
    return tuple(sorted(warmed))


def qsa_store_cache_rows(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    """Store fixed-width rows in a QSA cache without boolean indexing."""

    if not cache.is_cuda or not HAS_TRITON:
        raise RuntimeError("QSA CUDA cache stores require Triton")
    if cache.ndim != 4 or cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, width]")
    if not all(cache.shape):
        raise ValueError("QSA cache dimensions must be nonzero")
    if rows.ndim == 3:
        if rows.shape[1] != 1:
            raise ValueError("QSA cache rows must have one head")
        rows = rows[:, 0]
    if rows.shape != (slot_mapping.numel(), cache.shape[3]):
        raise ValueError("QSA cache rows and slots have incompatible shapes")
    if not rows.shape[0]:
        return
    _store_qsa_rows_kernel[(rows.shape[0],)](
        cache,
        slot_mapping,
        rows,
        cache.stride(0),
        cache.stride(1),
        cache.stride(3),
        rows.stride(0),
        rows.stride(1),
        rows.shape[0],
        cache.shape[0],
        PAGE_SIZE=cache.shape[1],
        WIDTH=cache.shape[3],
        BLOCK_D=triton.next_power_of_2(cache.shape[3]),
        num_warps=4,
    )


def qsa_compress_groups_with_ratio(
    raw_keys: torch.Tensor,  # this step's raw key rows [rows, 1, head_size]
    raw_positions: torch.Tensor,  # this step's positions [rows, 1, 3] int64
    compressor_state_cache: torch.Tensor,
    compressor_state_block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_start_loc: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_slots: torch.Tensor,
    compress_ratio: int,
    rope_cache: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool completed groups from the compressor-state ring and raw token rows."""

    if not raw_keys.is_cuda or not HAS_TRITON:
        raise RuntimeError("QSA CUDA compression requires Triton")
    rows = token_to_req.numel()
    if compress_ratio <= 0:
        raise ValueError("QSA compression ratio must be positive")
    if raw_keys.ndim != 3 or raw_keys.shape[:2] != (rows, 1):
        raise ValueError("QSA raw keys must be [rows, 1, head_size]")
    if raw_positions.shape != (rows, 1, 3) or raw_positions.dtype != torch.int64:
        raise ValueError("QSA raw positions must be [rows, 1, 3] int64")
    if logical_positions.shape != (rows,) or compressed_slots.shape != (rows,):
        raise ValueError("QSA compression metadata must match token rows")
    if compressor_state_cache.ndim != 4 or compressor_state_cache.shape[2] != 1:
        raise ValueError("QSA compressor-state cache has an invalid shape")
    if (
        # The ring is wider than one group so speculative rows cannot alias
        # onto the committed keys of the group still being collected.
        compressor_state_cache.shape[1] < compress_ratio
        or compressor_state_cache.shape[3] != raw_keys.shape[2]
        or compressor_state_cache.dtype != raw_keys.dtype
    ):
        raise ValueError(
            "QSA compressor-state cache does not match the compression layout"
        )
    if (
        compressor_state_block_table.ndim != 2
        or compressor_state_block_table.shape[1] < 1
    ):
        raise ValueError(
            "QSA compressor-state block table must contain one block per request"
        )
    if query_start_loc.ndim != 1 or query_start_loc.shape[0] < 2:
        raise ValueError("QSA query starts must contain a terminal offset")
    num_requests = query_start_loc.shape[0] - 1
    if compressor_state_block_table.shape[0] < num_requests:
        raise ValueError("QSA compressor-state block table has too few request rows")
    if rope_cache is not None and (
        rope_cache.ndim != 4
        or rope_cache.shape[:3] != compressor_state_cache.shape[:3]
        or rope_cache.shape[3] != 3
        or rope_cache.dtype != torch.int64
    ):
        raise ValueError("QSA packed position view has an invalid shape or dtype")
    if rows and (
        not all(compressor_state_cache.shape)
        or not all(compressor_state_block_table.shape)
    ):
        raise ValueError("QSA compressor-state cache and block table must be nonempty")
    pooled = torch.empty(
        (rows, 1, raw_keys.shape[2]),
        dtype=raw_keys.dtype,
        device=raw_keys.device,
    )
    first_positions = torch.empty((rows, 3), dtype=torch.int64, device=raw_keys.device)
    if not rows:
        return pooled, first_positions
    if rope_cache is None:
        rope_cache = compressor_state_cache
        load_rope_positions = False
    else:
        load_rope_positions = True
    _compress_qsa_groups_kernel[(rows,)](
        raw_keys,
        raw_positions,
        compressor_state_cache,
        rope_cache,
        compressor_state_block_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
        pooled,
        first_positions,
        raw_keys.stride(0),
        raw_keys.stride(2),
        raw_positions.stride(0),
        raw_positions.stride(2),
        compressor_state_cache.stride(0),
        compressor_state_cache.stride(1),
        compressor_state_cache.stride(3),
        rope_cache.stride(0),
        rope_cache.stride(1),
        rope_cache.stride(3),
        compressor_state_block_table.stride(0),
        pooled.stride(0),
        pooled.stride(2),
        first_positions.stride(0),
        first_positions.stride(1),
        rows,
        compressor_state_cache.shape[0],
        num_requests,
        COMPRESSOR_STATE_SIZE=compressor_state_cache.shape[1],
        COMPRESS_RATIO=compress_ratio,
        HEAD_DIM=raw_keys.shape[2],
        LOAD_ROPE_POSITIONS=load_rope_positions,
        BLOCK_D=triton.next_power_of_2(raw_keys.shape[2]),
        num_warps=4,
    )
    return pooled, first_positions


__all__ = [
    "qsa_compress_groups_with_ratio",
    "qsa_sparse_paged_attention",
    "qsa_store_cache_rows",
    "warmup_qsa_sparse_paged_attention",
]
