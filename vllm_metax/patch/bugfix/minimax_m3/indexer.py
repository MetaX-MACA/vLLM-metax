# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# This file backports upstream vLLM's fix for
# `vllm/models/minimax_m3/common/indexer.py`'s
# `MiniMaxM3IndexerTritonImpl.forward` -- the write-side counterpart of
# `sparse_attention.py`'s patch in this same directory; see that file's
# module-level note for the matching read-side fix.
#
# -----------------------------------------------
# Note: `self.topk_indices_buffer` (the shared, cudagraph-stable top-k
#       output buffer written here and read back in sparse_attention.py) is
#       allocated TOKEN-MAJOR by `vllm/models/minimax_m3/nvidia/model.py`:
#           torch.empty(padded_num_tokens, num_index_heads, topk_blocks, ...)
#       i.e. `[total_q, num_index_heads, topk]`, with a comment promising
#       "the attend transposes to [H, tokens, topk]". This impl's `out=`
#       slicing below (`buf[:, nd:, :]` etc.) assumes the OLD head-major
#       convention (`[num_index_heads, total_q, topk]` -- how
#       `amd/model.py` still allocates it) and never performs that promised
#       transpose.
#
#       Whenever `num_index_heads == 1` (true for MiniMax-M3 at any TP
#       degree where `sparse_num_index_heads // tp_size` rounds down to 0,
#       e.g. MiniMax-M3-W8A8 at TP=8: `max(1, 4 // 8) == 1`), this hides
#       almost perfectly: decode's `buf[:, :nd, :]` and prefill's
#       `buf[:, 0:, :]` (nd == 0, i.e. every pure-prefill step, including a
#       fresh engine's very first step) both clamp to the whole size-1 dim
#       and land on offset 0 by accident. The first time a batch mixes
#       decode with prefill (`nd > 0` -- from the 2nd engine step onward
#       under any real multi-request load), prefill's slice desyncs from
#       the true token offset, corrupting the topk indices the attention
#       kernel later uses to index the paged KV-cache block_table --
#       surfacing as a CUDA/MACA "illegal memory access". Confirmed
#       independently by reproducing this on MetaX C550; the same failure
#       is reported upstream on plain NVIDIA H800 (vllm-project/vllm#49147)
#       -- this is a vLLM correctness bug, not a MetaX- or Triton-backend
#       issue.
#
#       Fixed upstream in vllm-project/vllm#49149 (commit d1a8ba63d9d2,
#       merged 2026-07-25, closing vllm-project/vllm#49147) by transposing
#       the buffer to head-major right after fetching it, gated so ROCm
#       (whose buffer is already head-major) is left untouched. This file
#       backports that exact fix as a monkeypatch, since our vLLM install
#       is a released wheel we do not build/modify in place.
#
#       Verified NOT present in v0.24.0 (our base), v0.25.0, or v0.26.0 --
#       checked the buffer-fetch line in each tagged source directly. IS
#       present in v0.27.0. Drop this patch file once the vLLM dependency
#       is bumped to v0.27.0 or later.
#
# Affected versions: v0.24.0 (confirmed present through v0.26.0)
# -----------------------------------------------

import torch

from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform

from ._import_hooks import on_first_import

# Populated by `_apply_patch` once the real `indexer` module has imported.
MiniMaxM3IndexerMetadata = None
minimax_m3_index_decode = None
minimax_m3_index_score = None
minimax_m3_index_topk = None


def forward(self, index_query: torch.Tensor):
    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        return None, None  # profiling run; caches unbound
    index_md = attn_metadata[self.index_cache.prefix]
    assert isinstance(index_md, MiniMaxM3IndexerMetadata)
    num_tokens = index_md.num_actual_tokens
    nd = index_md.num_decode_tokens
    iq = index_query[:num_tokens].view(-1, self.num_index_heads, self.index_head_dim)
    kv = self.index_cache.kv_cache

    # Both sides write into the single shared persistent topk_indices_buffer
    # (decode at [:, :nd], prefill at [:, nd:]) and return views into it; the
    # kernels' out= writes out[:, :total_q]. None -> allocate fresh.
    buf = self.topk_indices_buffer
    # ┌------------------------  Metax Modification -------------------------┐
    # Backport of vllm-project/vllm#49149: transpose the token-major buffer
    # to head-major before slicing (skip on ROCm, whose buffer already is).
    buf_htk = buf if buf is None or current_platform.is_rocm() else buf.transpose(0, 1)
    # └------------------------  Metax Modification -------------------------┘
    decode_topk: torch.Tensor | None = None
    prefill_topk: torch.Tensor | None = None
    if index_md.num_decodes > 0:
        d = index_md.decode
        assert d is not None
        decode_topk = minimax_m3_index_decode(
            iq[:nd],
            kv,
            d.block_table,
            d.seq_lens,
            d.max_seq_len,
            self.topk_blocks,
            self.init_blocks,
            self.local_blocks,
            self.num_kv_heads,
            d.decode_query_len,
            d.max_decode_query_len,
            out=buf_htk,
        )
    if index_md.num_prefills > 0:
        p = index_md.prefill
        assert p is not None
        score = minimax_m3_index_score(
            iq[nd:],
            kv,
            p.block_table,
            p.cu_seqlens_q,
            p.seq_lens,
            p.context_lens,
            p.max_query_len,
            p.max_seq_len,
            self.num_kv_heads,
        )
        prefill_topk = minimax_m3_index_topk(
            score,
            p.cu_seqlens_q,
            p.context_lens,
            p.max_query_len,
            self.topk_blocks,
            self.init_blocks,
            self.local_blocks,
            out=buf_htk[:, nd:, :] if buf_htk is not None else None,
        )
    return decode_topk, prefill_topk


def _apply_patch(indexer_mod):
    global MiniMaxM3IndexerMetadata
    global minimax_m3_index_decode, minimax_m3_index_score, minimax_m3_index_topk
    MiniMaxM3IndexerMetadata = indexer_mod.MiniMaxM3IndexerMetadata
    minimax_m3_index_decode = indexer_mod.minimax_m3_index_decode
    minimax_m3_index_score = indexer_mod.minimax_m3_index_score
    minimax_m3_index_topk = indexer_mod.minimax_m3_index_topk

    indexer_mod.MiniMaxM3IndexerTritonImpl.forward = forward


on_first_import("vllm.models.minimax_m3.common.indexer", _apply_patch)
