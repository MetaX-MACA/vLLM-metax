# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# This file contains code copied from vLLM's
# `vllm/models/minimax_m3/common/ops/index_topk.py`.
#
# -----------------------------------------------
# Note: `_index_block_score_kernel` is launched without an explicit
#       `num_stages`, so Triton's automatic loop-pipelining picks a stage
#       count sized for NVIDIA's larger per-SM shared memory. On MACA
#       hardware (65536-byte shared-memory limit) that overflows:
#           triton.runtime.errors.OutOfResources: out of resource: shared
#           memory, Required: 82432, Hardware limit: 65536.
#       Pin `num_stages=1` at the call site to disable pipelining/double
#       buffering and stay within the MACA shared-memory limit. The kernel
#       itself is unchanged -- only the launch config differs -- so it is
#       imported rather than redefined.
#
#       `vllm/models/minimax_m3/common/indexer.py` does
#       `from ...index_topk import minimax_m3_index_score`, which copies
#       the name into its OWN module namespace at import time. Overwriting
#       the attribute on the `index_topk`/`ops` modules alone does not
#       affect that already-bound copy, so `indexer.py`'s binding must be
#       patched directly too -- see `_import_hooks.on_first_import`'s
#       docstring/module note for why this is done lazily rather than via a
#       plain eager `import` here.
#
#       `_decode_index_score_kernel` has the same MetaX MMA-encoder problem
#       as `sparse_attn.py`'s kernels (see that patch's notes): its
#       `tl.dot(k, q)` produces an [N, HQ] tile where
#       ``HQ = num_idx_heads * BLOCK_SIZE_Q``. M3 uses a single shared index
#       head (``num_idx_heads == 1``) and non-speculative decode uses
#       ``max_decode_query_len == 1``, so HQ defaults to 1 -- far below
#       MetaX's 16-minimum MMA operand tile, tripping the same
#       "tm/tn and tk not meet condition" assertion. Floor `BLOCK_SIZE_Q` (in
#       the `minimax_m3_index_decode` wrapper) so HQ >= 16; the extra padded
#       query slots are already masked out (`q_mask`) on both the q load and
#       the score store in the (unmodified) kernel, so this is safe. The
#       kernel itself is reused unchanged -- only the wrapper's launch
#       config differs.
#
# Affected versions: v0.24.0
# -----------------------------------------------

import sys

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.math_utils import round_up

from ._import_hooks import on_first_import

# Populated by `_apply_patch` once the real `index_topk` module has been
# imported (see the bottom of this file). `minimax_m3_index_score` /
# `minimax_m3_index_decode` below read these as plain module globals, which
# Python resolves at CALL time, not at def time -- so as long as
# `_apply_patch` has run before either function is ever actually invoked
# (guaranteed: nothing can call them before importing this module, and that
# import is exactly what triggers `_apply_patch`), this is safe.
SPARSE_BLOCK_SIZE = None
_index_block_score_kernel = None
_decode_index_score_kernel = None
_topk_index_partial_kernel = None
_topk_index_merge_kernel = None


@torch.no_grad()
def minimax_m3_index_score(
    idx_q: torch.Tensor,  # [total_q, num_idx_heads, head_dim]
    index_kv_cache: torch.Tensor,  # [num_blocks, 128, head_dim]
    block_table: torch.Tensor,  # [batch, max_blocks]
    cu_seqlens_q: torch.Tensor,  # [batch+1] int32
    seq_lens: torch.Tensor,  # [batch] int32
    prefix_lens: torch.Tensor,  # [batch] int32
    max_query_len: int,
    max_seq_len: int,
    num_kv_heads: int,
) -> torch.Tensor:
    """Compute per-token index scores for each visible sparse block.

    Returns score [num_kv_heads, total_q, max_block], where each score is the
    max over a 128-token index-K block. M3 has num_idx_heads == num_kv_heads.
    """
    total_q, num_idx_heads, head_dim = idx_q.shape
    assert num_idx_heads == num_kv_heads, (
        "M3 expects num_idx_heads == num_kv_heads (no topk index reduce)"
    )
    batch = cu_seqlens_q.shape[0] - 1
    max_block = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)

    # Keep score strides 16-divisible to avoid Triton recompiles.
    score_block_stride = round_up(max_block, 16)
    score = torch.empty(
        (num_idx_heads, total_q, score_block_stride),
        dtype=torch.float32,
        device=idx_q.device,
    )
    BLOCK_SIZE_Q = 64
    grid_score = (triton.cdiv(max_query_len, BLOCK_SIZE_Q), batch * num_idx_heads)
    _index_block_score_kernel[grid_score](
        idx_q,
        index_kv_cache,
        score,
        block_table,
        cu_seqlens_q,
        seq_lens,
        prefix_lens,
        num_idx_heads,
        head_dim,
        idx_q.stride(0),
        idx_q.stride(1),
        idx_q.stride(2),
        index_kv_cache.stride(0),
        index_kv_cache.stride(1),
        index_kv_cache.stride(2),
        score.stride(0),
        score.stride(1),
        score.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        # ┌------------------------  Metax Modification -------------------------┐
        num_stages=1,
        # └------------------------  Metax Modification -------------------------┘
    )
    return score


@torch.no_grad()
def minimax_m3_index_decode(
    idx_q: torch.Tensor,  # [total_q, num_idx_heads, head_dim]
    index_kv_cache: torch.Tensor,  # [num_blocks, 128, head_dim]
    block_table: torch.Tensor,  # [num_reqs, max_blocks]
    seq_lens: torch.Tensor,  # [num_reqs] int32
    max_seq_len: int,
    topk: int,
    init_blocks: int,
    local_blocks: int,
    num_kv_heads: int,
    decode_query_len: int,
    max_decode_query_len: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode index block-score + top-k, both split-K (cudagraph-safe).

    Returns topk_idx [num_kv_heads, total_q, topk] (0-indexed block ids, -1 pad).
    When ``out`` ([num_kv_heads, >=total_q, topk]) is given, writes into
    ``out[:, :total_q, :]`` (stable address for cudagraph) instead of allocating.
    """
    total_q, num_idx_heads, head_dim = idx_q.shape
    assert num_idx_heads == num_kv_heads, (
        "M3 expects num_idx_heads == num_kv_heads (no topk index reduce)"
    )
    assert decode_query_len <= max_decode_query_len
    assert total_q == seq_lens.shape[0] * decode_query_len
    batch = total_q
    max_block = triton.cdiv(max_seq_len, SPARSE_BLOCK_SIZE)
    use_pdl = current_platform.is_arch_support_pdl()
    # `launch_pdl` is a Triton runtime kwarg only some backends accept (CUDA
    # SM9+); this ROCm Triton rejects it even when False ("Keyword argument
    # launch_pdl was specified but unrecognised"). Only pass it when PDL is
    # actually supported -- on ROCm use_pdl is always False, so it's omitted.
    pdl_kwargs: dict[str, bool | int] = {}
    if use_pdl:
        pdl_kwargs.update({"launch_pdl": True})
    # TP=1 spec decode scores a wide 4-head x 4-position query tile per K block;
    # reduce stages to ease memory/register pressure. Keep no-spec and TP=4
    # single-head codegen unchanged.
    score_kwargs = pdl_kwargs.copy()
    if num_idx_heads > 1 and max_decode_query_len > 1:
        score_kwargs.update({"num_warps": 4, "num_stages": 2})
    # ┌------------------------  Metax Modification -------------------------┐
    # Same MACA 64-KB shared-memory ceiling as `_index_block_score_kernel`.
    # The BLOCK_SIZE_Q floor below widens the q/k/v tiles enough that
    # Triton's default multi-stage pipelining (double buffering) overflows it
    # ("Required: 69888, Hardware limit: 65536"); force single-buffered.
    # Set last so it always wins over the num_stages=2 branch above.
    score_kwargs["num_stages"] = 1
    # └------------------------  Metax Modification -------------------------┘

    # Keep score strides 16-divisible to avoid Triton recompiles.
    score_block_stride = round_up(max_block, 16)
    score = torch.empty(
        (num_idx_heads, total_q, score_block_stride),
        dtype=torch.float32,
        device=idx_q.device,
    )
    # split-K over seq blocks; chunk count depends only on shape constants so
    # the grid is fixed within a cuda graph.
    TARGET_GRID = 512
    MAX_NUM_KV_CHUNKS = 256
    # Use the configured max decode length to avoid Triton recompiles when
    # switching between qlen=1 and spec-decode verification batches.
    BLOCK_SIZE_Q = triton.next_power_of_2(max_decode_query_len)
    # ┌------------------------  Metax Modification -------------------------┐
    # Floor BLOCK_SIZE_HQ (= num_idx_heads * BLOCK_SIZE_Q) at 16 -- see the
    # module-level note. Padded query slots beyond decode_query_len are
    # already masked out (`q_mask`) inside the (unmodified) kernel.
    while num_idx_heads * BLOCK_SIZE_Q < 16:
        BLOCK_SIZE_Q *= 2
    # └------------------------  Metax Modification -------------------------┘
    score_ctas_per_chunk = seq_lens.shape[0]
    target = max(
        1,
        min(MAX_NUM_KV_CHUNKS, TARGET_GRID // max(1, score_ctas_per_chunk)),
    )
    num_kv_chunks = 1 << (target.bit_length() - 1)
    grid_score = (seq_lens.shape[0], num_kv_chunks)
    _decode_index_score_kernel[grid_score](
        idx_q,
        index_kv_cache,
        score,
        block_table,
        seq_lens,
        num_idx_heads,
        head_dim,
        init_blocks,
        local_blocks,
        decode_query_len,
        idx_q.stride(0),
        idx_q.stride(1),
        idx_q.stride(2),
        index_kv_cache.stride(0),
        index_kv_cache.stride(1),
        index_kv_cache.stride(2),
        score.stride(0),
        score.stride(1),
        score.stride(2),
        block_table.stride(0),
        BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        BLOCK_SIZE_Q=BLOCK_SIZE_Q,
        num_kv_chunks=num_kv_chunks,
        USE_PDL=use_pdl,
        **score_kwargs,
    )

    if out is not None:
        topk_idx = out[:, :total_q, :]
    else:
        topk_idx = torch.empty(
            (num_idx_heads, total_q, topk),
            dtype=torch.int32,
            device=idx_q.device,
        )
    # Chunk count is shape-constant (cudagraph-safe), capped so the merge sorts
    # pow2(num_topk_chunks * pow2(topk)) candidates.
    TOPK_TARGET_GRID = 64
    MAX_NUM_TOPK_CHUNKS = 16
    topk_target = max(
        1, min(MAX_NUM_TOPK_CHUNKS, TOPK_TARGET_GRID // max(1, batch * num_idx_heads))
    )
    num_topk_chunks = 1 << (topk_target.bit_length() - 1)
    block_size_t = triton.next_power_of_2(topk)
    chunk_blocks = (max_block + num_topk_chunks - 1) // num_topk_chunks
    topk_score_partial = torch.empty(
        num_topk_chunks,
        num_idx_heads,
        batch,
        block_size_t,
        dtype=torch.float32,
        device=idx_q.device,
    )
    topk_idx_partial = torch.empty(
        num_topk_chunks,
        num_idx_heads,
        batch,
        block_size_t,
        dtype=torch.int32,
        device=idx_q.device,
    )
    _topk_index_partial_kernel[(batch, num_idx_heads, num_topk_chunks)](
        score,
        topk_score_partial,
        topk_idx_partial,
        seq_lens,
        SPARSE_BLOCK_SIZE,
        topk,
        chunk_blocks,
        decode_query_len,
        score.stride(0),
        score.stride(1),
        score.stride(2),
        topk_score_partial.stride(0),
        topk_score_partial.stride(1),
        topk_score_partial.stride(2),
        topk_score_partial.stride(3),
        topk_idx_partial.stride(0),
        topk_idx_partial.stride(1),
        topk_idx_partial.stride(2),
        topk_idx_partial.stride(3),
        USE_PDL=use_pdl,
        **pdl_kwargs,
    )
    _topk_index_merge_kernel[(batch, num_idx_heads)](
        topk_score_partial,
        topk_idx_partial,
        topk_idx,
        seq_lens,
        SPARSE_BLOCK_SIZE,
        topk,
        decode_query_len,
        topk_score_partial.stride(0),
        topk_score_partial.stride(1),
        topk_score_partial.stride(2),
        topk_score_partial.stride(3),
        topk_idx_partial.stride(0),
        topk_idx_partial.stride(1),
        topk_idx_partial.stride(2),
        topk_idx_partial.stride(3),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        num_topk_chunks=num_topk_chunks,
        USE_PDL=use_pdl,
        **pdl_kwargs,
    )
    return topk_idx


def _apply_patch(index_topk_mod):
    global SPARSE_BLOCK_SIZE, _index_block_score_kernel
    global \
        _decode_index_score_kernel, \
        _topk_index_partial_kernel, \
        _topk_index_merge_kernel
    SPARSE_BLOCK_SIZE = index_topk_mod.SPARSE_BLOCK_SIZE
    _index_block_score_kernel = index_topk_mod._index_block_score_kernel
    _decode_index_score_kernel = index_topk_mod._decode_index_score_kernel
    _topk_index_partial_kernel = index_topk_mod._topk_index_partial_kernel
    _topk_index_merge_kernel = index_topk_mod._topk_index_merge_kernel

    index_topk_mod.minimax_m3_index_score = minimax_m3_index_score
    index_topk_mod.minimax_m3_index_decode = minimax_m3_index_decode

    # `ops/__init__.py` re-exports the original functions under its own name
    # -- since this callback runs before control returns to whatever import
    # statement pulled in `index_topk_mod` (see `_import_hooks.py`), that
    # re-export (if it hasn't happened yet) will already see the patched
    # attributes; this direct assignment is just a belt-and-suspenders
    # backstop for the case where `ops` was already fully imported earlier.
    import vllm.models.minimax_m3.common.ops as _ops

    _ops.minimax_m3_index_score = minimax_m3_index_score
    _ops.minimax_m3_index_decode = minimax_m3_index_decode

    # `indexer.py` and `nvidia/indexer_msa.py` also do their own
    # `from ...index_topk import ...`. If either is *already* imported (i.e.
    # this callback is firing on an `index_topk` that was imported before
    # this bugfix even loaded), their copy of the name is already stale and
    # must be patched directly -- same reasoning as `_ops` above. If neither
    # is imported yet, there is nothing to do: whichever imports
    # `index_topk` later will see these already-patched attributes.
    for _mod_name in (
        "vllm.models.minimax_m3.common.indexer",
        "vllm.models.minimax_m3.nvidia.indexer_msa",
    ):
        _mod = sys.modules.get(_mod_name)
        if _mod is not None:
            _mod.minimax_m3_index_score = minimax_m3_index_score
            _mod.minimax_m3_index_decode = minimax_m3_index_decode


on_first_import("vllm.models.minimax_m3.common.ops.index_topk", _apply_patch)
