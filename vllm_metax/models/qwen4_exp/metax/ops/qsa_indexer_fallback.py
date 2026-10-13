# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Memory-bounded QSA index-selection fallback for MACA devices."""

from __future__ import annotations

import math
import os

import torch
from vllm.triton_utils import tl, triton

_USE_OPTIMIZED = bool(int(os.getenv("VLLM_METAX_QSA_OPTIMIZED", "1")))
_USE_TRITON_TOPK = bool(int(os.getenv("VLLM_METAX_QSA_TRITON_TOPK", "1")))
_USE_SCORE_EARLY_EXIT = bool(int(os.getenv("VLLM_METAX_QSA_SCORE_EARLY_EXIT", "1")))


@triton.jit
def _qsa_index_score_kernel(
    q_ptr,
    k_ptr,
    table_ptr,
    req_ptr,
    visible_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_table_req,
    stride_logits_row,
    num_columns,
    PAGE_SIZE: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    CACHE_PAGES: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EARLY_EXIT: tl.constexpr,
):
    """MACA-friendly QSA score kernel which avoids a narrow-N MMA."""
    row = tl.program_id(0)
    tile_start = tl.program_id(1) * BLOCK_N
    cols = tile_start + tl.arange(0, BLOCK_N)
    req = tl.load(req_ptr + row)
    visible = tl.load(visible_ptr + row)
    if EARLY_EXIT:
        # A tile fully beyond the visible prefix only emits -inf stores;
        # skip its page-table and K loads and the per-head dot loop.
        # (Same shortcut as the fused ops/qsa_indexer.py kernel.)
        if tile_start >= visible:
            neg_inf = tl.full((BLOCK_N,), -float("inf"), tl.float32)
            tl.store(
                logits_ptr + row * stride_logits_row + cols,
                neg_inf,
                cols < num_columns,
            )
            return
    dims = tl.arange(0, BLOCK_D)
    valid_col = (cols < num_columns) & (cols < visible)
    logical_page = tl.minimum(cols // PAGE_SIZE, TABLE_WIDTH - 1)
    page_offset = cols % PAGE_SIZE
    physical_page = tl.load(
        table_ptr + req * stride_table_req + logical_page,
        mask=valid_col,
        other=0,
    ).to(tl.int64)
    valid_col &= (physical_page >= 0) & (physical_page < CACHE_PAGES)
    keys = tl.load(
        k_ptr
        + physical_page[:, None] * stride_k_block
        + page_offset[:, None] * stride_k_token
        + dims[None, :],
        mask=valid_col[:, None] & (dims[None, :] < HEAD_DIM),
        other=0.0,
        eviction_policy="evict_first",
    ).to(tl.float32)
    score = tl.zeros((BLOCK_N,), tl.float32)
    for head in range(NUM_HEADS):
        query = tl.load(
            q_ptr + row * stride_q_row + head * stride_q_head + dims,
            mask=dims < HEAD_DIM,
            other=0.0,
        ).to(tl.float32)
        dot = tl.sum(keys * query[None, :], axis=1)
        score += tl.maximum(dot, 0.0)
    score *= 1.0 / (HEAD_DIM**0.5)
    tl.store(
        logits_ptr + row * stride_logits_row + cols,
        tl.where(valid_col, score, -float("inf")),
        cols < num_columns,
    )


@triton.jit
def _store_topk_indices(
    indices,
    values,
    output,
    stride_in,
    stride_out_row,
    stride_out_col,
    WIDTH: tl.constexpr,
    SELECTED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    active = col < SELECTED
    index = tl.load(indices + row * stride_in + col, active, other=-1)
    score = tl.load(values + row * stride_in + col, active, other=-float("inf"))
    # Invalid/missing selections are sentinels, never future cache positions.
    index = tl.where(active & (score != -float("inf")), index, -1)
    tl.store(output + row * stride_out_row + col * stride_out_col, index, col < WIDTH)


@triton.jit
def _qsa_key32(score):
    """Order-preserving fp32 -> signed int32 key (ascending float order).

    Positive floats (bit31 clear) keep their bits; negative floats xor with
    0x7FFFFFFF, which flips the magnitude bits and leaves bit31 set, so all
    negative floats map below all positive ones while keeping their order.
    Compared as plain int32, so no 64-bit integer ops are needed.
    """
    b = score.to(tl.int32, bitcast=True)
    return tl.where(b >= 0, b, b ^ 2147483647)


@triton.jit
def _qsa_persistent_topk_kernel(
    logits_ptr,
    visible_ptr,
    out_ptr,
    stride_logits_row,
    stride_out_row,
    stride_out_col,
    num_columns,
    SELECTED: tl.constexpr,
    WIDTH: tl.constexpr,
    WIDTH_P2: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    """Fused persistent top-k over the visible prefix of each logits row.

    Replaces the ``torch.topk`` + ``_store_topk_indices`` chain, whose MACA
    ATen mbtopk implementation costs ~295us/call at [96, 5600] / k=512
    (~14 kernels including two ~68us workspace memsets and a sorted=True
    radix sort). One program owns one row and makes a single sweep:

    1. A 32-step binary search over the order-preserving int32 key of the
       fp32 scores locates the k-th largest key. When the visible prefix
       fits in one tile (the common decode case) the keys are loaded once
       and stay in registers across the search; wider rows re-read tiles
       from L2. Only the ``visible`` leading columns are scanned.
    2. One deterministic compaction pass (``tl.cumsum`` in column order)
       writes the selected block indices directly to ``out_ptr``. Every
       score strictly above the k-th key is kept; threshold ties are
       admitted in column order only until the k slots fill, so a large
       score at a late column is never displaced by a tie at an early one.

    The -1 sentinel contract of ``_store_topk_indices`` is preserved:
    holes (invalid pages, scored -inf) that only win slots when fewer
    than k finite scores exist are stored as -1, and trailing unfilled
    slots are -1. The grid is static per capture shape (no host sync,
    no dynamic allocation), so the launch is CUDA-graph safe.
    """
    row = tl.program_id(0)
    visible = tl.minimum(tl.load(visible_ptr + row).to(tl.int32), num_columns)
    k = tl.minimum(SELECTED, visible)

    # Overflow-safe upper midpoint: ceil((lo + hi) / 2) evaluated as
    # (lo & hi) + ((lo ^ hi) >> 1) + ((lo ^ hi) & 1), valid for the full
    # signed int32 domain including negatives. (A three-mid variant halves
    # the rounds but measures slower: every tl.sum reduction serializes on
    # the barrier, so fewer reductions win over fewer dependent steps.)
    if visible <= BLOCK_W:
        cols = tl.arange(0, BLOCK_W)
        live = cols < visible
        score = tl.load(
            logits_ptr + row * stride_logits_row + cols,
            mask=live,
            other=-float("inf"),
        )
        key = _qsa_key32(score)
        lo = -2147483647 - 1
        hi = 2147483647
        for _ in range(32):
            mid = (lo & hi) + ((lo ^ hi) >> 1) + ((lo ^ hi) & 1)
            cnt = tl.sum((live & (key >= mid)).to(tl.int32))
            take = cnt >= k
            lo = tl.where(take, mid, lo)
            hi = tl.where(take, hi, mid - 1)
        # Compaction: every candidate strictly above the k-th key is kept,
        # and only enough threshold ties to fill the remaining k - n_gt
        # slots are admitted, in column order. Truncating all candidates
        # by column order alone could drop a larger score at a late column
        # in favour of a threshold tie at an early one.
        gt = live & (key > lo)
        eq = live & (key == lo)
        n_gt = tl.sum(gt.to(tl.int32))
        eq_rank = tl.cumsum(eq.to(tl.int32), axis=0) - 1
        keep = gt | (eq & (eq_rank < k - n_gt))
        pos = tl.cumsum(keep.to(tl.int32), axis=0) - 1
        selected = tl.where(score > -float("inf"), cols, -1)
        tl.store(
            out_ptr + row * stride_out_row + pos * stride_out_col,
            selected,
            mask=keep,
        )
        base = tl.sum(keep.to(tl.int32))
    else:
        lo = -2147483647 - 1
        hi = 2147483647
        for _ in range(32):
            mid = (lo & hi) + ((lo ^ hi) >> 1) + ((lo ^ hi) & 1)
            cnt = 0
            for tile in range(0, visible, BLOCK_W):
                cols = tile + tl.arange(0, BLOCK_W)
                live = cols < visible
                score = tl.load(
                    logits_ptr + row * stride_logits_row + cols,
                    mask=live,
                    other=-float("inf"),
                )
                key = _qsa_key32(score)
                cnt += tl.sum((live & (key >= mid)).to(tl.int32))
            take = cnt >= k
            lo = tl.where(take, mid, lo)
            hi = tl.where(take, hi, mid - 1)

        # Same tie rule as the fast path, but the candidates are spread
        # over tiles: first count strictly-above and tied candidates, then
        # admit ties by their global column-order rank until k slots fill.
        n_gt = 0
        n_eq = 0
        for tile in range(0, visible, BLOCK_W):
            cols = tile + tl.arange(0, BLOCK_W)
            live = cols < visible
            score = tl.load(
                logits_ptr + row * stride_logits_row + cols,
                mask=live,
                other=-float("inf"),
            )
            key = _qsa_key32(score)
            n_gt += tl.sum((live & (key > lo)).to(tl.int32))
            n_eq += tl.sum((live & (key == lo)).to(tl.int32))
        admit = k - n_gt

        base = 0
        eq_base = 0
        for tile in range(0, visible, BLOCK_W):
            cols = tile + tl.arange(0, BLOCK_W)
            live = cols < visible
            score = tl.load(
                logits_ptr + row * stride_logits_row + cols,
                mask=live,
                other=-float("inf"),
            )
            key = _qsa_key32(score)
            gt = live & (key > lo)
            eq = live & (key == lo)
            eq_rank = eq_base + tl.cumsum(eq.to(tl.int32), axis=0) - 1
            keep = gt | (eq & (eq_rank < admit))
            pos = base + tl.cumsum(keep.to(tl.int32), axis=0) - 1
            # Holes score -inf and never hold a real cache position.
            selected = tl.where(score > -float("inf"), cols, -1)
            tl.store(
                out_ptr + row * stride_out_row + pos * stride_out_col,
                selected,
                mask=keep,
            )
            base += tl.sum(keep.to(tl.int32))
            eq_base += tl.sum(eq.to(tl.int32))

    # Trailing unfilled slots keep the -1 sentinel, as _store_topk_indices.
    tail = tl.arange(0, WIDTH_P2)
    neg = tl.full((WIDTH_P2,), -1, tl.int32)
    tl.store(
        out_ptr + row * stride_out_row + tail * stride_out_col,
        neg,
        mask=(tail >= base) & (tail < WIDTH),
    )


def _select_blocks(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    visible_blocks: torch.Tensor,
    block_indices: torch.Tensor,
    *,
    workspace_bytes: int = 128 * 1024 * 1024,
    max_columns: int | None = None,
) -> None:
    """Select QSA blocks using bounded, device-only PyTorch operations."""
    rows = q.shape[0]
    columns = page_table.shape[1] * k_cache.shape[1]
    if max_columns is not None:
        columns = min(columns, max_columns)
    width = block_indices.shape[1]
    if rows == 0 or width == 0:
        return
    if columns == 0:
        block_indices.fill_(-1)
        return

    # The Triton kernel reads paged keys directly and only materializes logits.
    # Chunk logits to a fixed budget; there is no gathered K tensor.
    bytes_per_row = max(columns * 4, 1)
    rows_per_chunk = max(1, min(rows, workspace_bytes // bytes_per_row))
    for start in range(0, rows, rows_per_chunk):
        end = min(start + rows_per_chunk, rows)
        logits = torch.empty(
            (end - start, columns), dtype=torch.float32, device=q.device
        )
        # Launch shape from a device-time sweep at the decode logits shape
        # ([96, 5600], visible 1024..5600): BLOCK_N=32 with one warp beats
        # BLOCK_N=16 with four on MACA at every visibility (46 vs 137 us at
        # visible=1024, 153 vs 231 us at visible=5600) because the tl.sum
        # reductions never cross a warp barrier.
        block_n = 32
        block_d = triton.next_power_of_2(q.shape[-1])
        _qsa_index_score_kernel[(end - start, triton.cdiv(columns, block_n))](
            q[start:end],
            k_cache,
            page_table,
            token_to_req[start:end],
            visible_blocks[start:end],
            logits,
            *q.stride()[:-1],
            *k_cache.stride()[:2],
            *page_table.stride()[:-1],
            *logits.stride()[:-1],
            columns,
            PAGE_SIZE=k_cache.shape[1],
            TABLE_WIDTH=page_table.shape[1],
            CACHE_PAGES=k_cache.shape[0],
            NUM_HEADS=q.shape[1],
            HEAD_DIM=q.shape[2],
            BLOCK_N=block_n,
            BLOCK_D=block_d,
            EARLY_EXIT=_USE_SCORE_EARLY_EXIT,
            num_warps=1,
        )
        selected = min(width, columns)
        if _USE_TRITON_TOPK:
            _qsa_persistent_topk_kernel[(end - start,)](
                logits,
                visible_blocks[start:end],
                block_indices[start:end],
                logits.stride(0),
                block_indices.stride(0),
                block_indices.stride(1),
                columns,
                SELECTED=selected,
                WIDTH=width,
                WIDTH_P2=triton.next_power_of_2(width),
                BLOCK_W=1024,
                num_warps=4,
            )
        else:
            values, indices = torch.topk(logits, selected, dim=-1)
            _store_topk_indices[(end - start, triton.cdiv(width, 256))](
                indices,
                values,
                block_indices[start:end],
                indices.stride(0),
                block_indices.stride(0),
                block_indices.stride(1),
                WIDTH=width,
                SELECTED=selected,
                BLOCK=256,
            )


def qsa_select_paged_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    visible_blocks: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    decode_query_len: int,
    block_indices: torch.Tensor,
) -> None:
    """Portable replacement for the NVIDIA-oriented decode selector."""
    if token_topk % compress_ratio:
        raise ValueError("QSA token top-k must be divisible by compression ratio")
    num_requests = q.shape[0] // decode_query_len
    token_to_req = torch.arange(num_requests, device=q.device).repeat_interleave(
        decode_query_len
    )
    _select_blocks(q, k_cache, page_table, token_to_req, visible_blocks, block_indices)


def qsa_select_paged_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    visible_blocks: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    max_query_len: int,
    block_indices: torch.Tensor,
    max_seq_len: int,
) -> None:
    """Portable replacement for the NVIDIA-oriented prefill selector."""
    del max_query_len
    if token_topk % compress_ratio:
        raise ValueError("QSA token top-k must be divisible by compression ratio")
    relative_starts = query_start_loc - query_start_loc[0]
    rows = torch.arange(q.shape[0], device=q.device, dtype=relative_starts.dtype)
    token_to_req = torch.bucketize(rows, relative_starts[1:], right=True)
    _select_blocks(
        q,
        k_cache,
        page_table,
        token_to_req,
        visible_blocks,
        block_indices,
        max_columns=(
            max(1, math.ceil(max_seq_len / compress_ratio)) if _USE_OPTIMIZED else None
        ),
    )


__all__ = ["qsa_select_paged_decode", "qsa_select_paged_prefill"]
