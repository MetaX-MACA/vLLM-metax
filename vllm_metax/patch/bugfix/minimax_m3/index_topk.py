# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: `_index_block_score_kernel` (prefill index score) is launched without
#       an explicit `num_stages`, so Triton's automatic loop-pipelining picks
#       a stage count sized for NVIDIA's larger per-SM shared memory. On MACA
#       hardware (65536-byte shared-memory limit) that overflows.
#       `_decode_index_score_kernel` (decode index score) hits a separate
#       MetaX MMA-encoder minimum-operand-tile assertion whenever its
#       BLOCK_SIZE_HQ (= num_idx_heads * BLOCK_SIZE_Q) tile falls below
#       MetaX's 16-minimum. Both now surface at kernel COMPILE time as
#       `RuntimeError: PassManager::run failed` (make_ttgir) -- confirmed
#       live on MetaX C550 running MiniMax-M3-W8A8 @ TP=8, where
#       num_idx_heads == 1 and max_decode_query_len == 1 (no speculative
#       decoding), so BLOCK_SIZE_HQ defaults to 1 (far below 16).
#
#       Both kernel bodies (`_index_block_score_kernel`,
#       `_decode_index_score_kernel`) are byte-identical to v0.24.0's; only
#       the Python wrapper was refactored (v0.26.0 split
#       `minimax_m3_index_decode` into a score-only
#       `minimax_m3_index_decode_score` + a thin top-k-selecting
#       `minimax_m3_index_decode`, to allow sharing a unified score buffer
#       with the prefill side). This ports the same fix vllm_metax already
#       carried for v0.24.0 onto the new wrapper names: pin `num_stages=1`
#       to disable double-buffering (fits under the 64-KB ceiling), and
#       floor `BLOCK_SIZE_Q` so `num_idx_heads * BLOCK_SIZE_Q >= 16`.
#
# Affected versions: All versions (kernel bodies unchanged upstream;
#       confirmed still failing on v0.26.0 with the refactored wrapper).
#       Root‑cause: insufficient shared‑memory size on C500‑series hardware.
#
# Remove at: MetaX Triton backend (mcTriton) drops the 16-minimum MMA
#       operand-tile requirement and the 64-KB shared-memory ceiling stops
#       binding for these launch configs, or upstream widens them itself.
# -----------------------------------------------------------------------------
"""MetaX MACA shared-memory / MMA-tile fix for the index-score kernels.

Ports vllm_metax's v0.24.0 fix for `minimax_m3_index_score` and
`minimax_m3_index_decode_score` (formerly `minimax_m3_index_decode`) onto
v0.26.0's refactored wrappers. Kernel bodies are unmodified -- only the
launch configuration differs, so the kernels are imported rather than
redefined.
"""

import torch

from vllm.models.minimax_m3.common.ops.index_topk import (
    SPARSE_BLOCK_SIZE,
    _decode_index_score_kernel,
    _index_block_score_kernel,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.math_utils import round_up

from vllm_metax.patch.utils import patch


@patch("vllm.models.minimax_m3.common.ops.index_topk")
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
        # /-------------------- MetaX Modification --------------------\
        num_stages=1,
        # \-------------------- MetaX Modification --------------------/
    )
    return score


@patch("vllm.models.minimax_m3.common.ops.index_topk")
@torch.no_grad()
def minimax_m3_index_decode_score(
    idx_q: torch.Tensor,  # [total_q, num_idx_heads, head_dim]
    index_kv_cache: torch.Tensor,  # [num_blocks, 128, head_dim]
    block_table: torch.Tensor,  # [num_reqs, max_blocks]
    seq_lens: torch.Tensor,  # [num_reqs] int32
    max_seq_len: int,
    init_blocks: int,
    local_blocks: int,
    num_kv_heads: int,
    decode_query_len: int,
    max_decode_query_len: int,
    score_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode index block-score (split-K, cudagraph-safe); no top-k.

    Returns score [num_kv_heads, total_q, >=max_block] (fp32; init/local blocks
    forced to 1e30/1e29). When ``score_out`` is given the scores are written into
    it (read/written by strides, so a transposed view of a unified buffer is
    accepted) instead of a fresh tensor -- used to share a unified score buffer
    with the prefill side and run a single top-k over both.
    """
    total_q, num_idx_heads, head_dim = idx_q.shape
    assert num_idx_heads == num_kv_heads, (
        "M3 expects num_idx_heads == num_kv_heads (no topk index reduce)"
    )
    assert decode_query_len <= max_decode_query_len
    assert total_q == seq_lens.shape[0] * decode_query_len
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
    # /-------------------- MetaX Modification --------------------\
    # Same MACA 64-KB shared-memory ceiling as `_index_block_score_kernel`.
    # Set last so it always wins over the num_stages=2 branch above.
    score_kwargs["num_stages"] = 1
    # \-------------------- MetaX Modification --------------------/

    if score_out is not None:
        score = score_out
    else:
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
    # /-------------------- MetaX Modification --------------------\
    # Floor BLOCK_SIZE_HQ (= num_idx_heads * BLOCK_SIZE_Q) at 16 -- MetaX's
    # MMA encoder requires operand tiles >= 16. Padded query slots beyond
    # decode_query_len are already masked out (q_mask) inside the
    # (unmodified) kernel.
    while num_idx_heads * BLOCK_SIZE_Q < 16:
        BLOCK_SIZE_Q *= 2
    # \-------------------- MetaX Modification --------------------/
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
    return score
