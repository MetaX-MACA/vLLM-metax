# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# This file backports upstream vLLM's fix for
# `vllm/models/minimax_m3/common/sparse_attention.py`'s
# `MiniMaxM3SparseTritonImpl.forward` -- the read-side counterpart of
# `indexer.py`'s patch in this same directory; see that file's module-level
# note for the full root-cause explanation (token-major topk_indices_buffer
# vs. this impl's head-major slicing assumption, why num_index_heads == 1
# hides it until the first decode+prefill mixed batch, and the link to
# vllm-project/vllm#49147 / #49149).
#
# Affected versions: v0.24.0 (confirmed present through v0.26.0; fixed in
# v0.27.0 -- drop this patch file once the vLLM dependency is bumped to
# v0.27.0 or later).
# -----------------------------------------------

from vllm.forward_context import get_forward_context
from vllm.platforms import current_platform

from ._import_hooks import on_first_import

# Populated by `_apply_patch` once the real `sparse_attention` module has imported.
MiniMaxM3SparseMetadata = None
minimax_m3_sparse_attn = None
minimax_m3_sparse_attn_decode = None


def forward(self, layer, query, kv_cache, output):
    attn_metadata = get_forward_context().attn_metadata
    if not isinstance(attn_metadata, dict):
        return output  # profiling run; caches unbound
    main_md = attn_metadata[layer.layer_name]  # type: ignore[attr-defined]
    assert isinstance(main_md, MiniMaxM3SparseMetadata)

    nd = main_md.num_decode_tokens
    num_tokens = main_md.num_actual_tokens
    # Indexer top-k from the shared buffer: decode [:, :nd], prefill [:, nd:].
    topk_buffer = layer.topk_indices_buffer  # type: ignore[attr-defined]
    assert topk_buffer is not None

    # ┌------------------------  Metax Modification -------------------------┐
    # Backport of vllm-project/vllm#49149: the shared topk_indices_buffer is
    # allocated token-major by nvidia/model.py; transpose it to the
    # head-major view this impl's slicing below assumes (skip on ROCm,
    # whose buffer is already head-major).
    topk = (
        topk_buffer
        if current_platform.is_rocm()
        else topk_buffer[:num_tokens].transpose(0, 1)
    )
    # └------------------------  Metax Modification -------------------------┘
    assert topk is not None
    hd = self.head_size
    q = query[:num_tokens].view(-1, self.num_heads, hd)
    out = output[:num_tokens].view(-1, self.num_heads, hd)
    kv_cache = kv_cache.view(self.kv_cache_fp8_dtype) if self.use_fp8_kv else kv_cache

    # Decode [:nd]: split-K over the selected blocks (request-major chunks).
    if main_md.num_decodes > 0:
        d = main_md.decode
        assert d is not None
        minimax_m3_sparse_attn_decode(
            q[:nd],
            kv_cache,
            topk[:, :nd, :],
            d.block_table,
            d.seq_lens,
            self.num_kv_heads,
            self.scale,
            out[:nd],
            d.decode_query_len,
        )

    # Prefill [nd:]: cu_seqlens_q already rebased to 0.
    if main_md.num_prefills > 0:
        p = main_md.prefill
        assert p is not None
        minimax_m3_sparse_attn(
            q[nd:],
            kv_cache,
            topk[:, nd:num_tokens, :],
            p.block_table,
            p.cu_seqlens_q,
            p.seq_lens,
            p.context_lens,
            p.max_query_len,
            self.num_kv_heads,
            self.scale,
            out[nd:],
        )
    return output


def _apply_patch(sparse_attention_mod):
    global \
        MiniMaxM3SparseMetadata, \
        minimax_m3_sparse_attn, \
        minimax_m3_sparse_attn_decode
    MiniMaxM3SparseMetadata = sparse_attention_mod.MiniMaxM3SparseMetadata
    minimax_m3_sparse_attn = sparse_attention_mod.minimax_m3_sparse_attn
    minimax_m3_sparse_attn_decode = sparse_attention_mod.minimax_m3_sparse_attn_decode

    sparse_attention_mod.MiniMaxM3SparseTritonImpl.forward = forward


on_first_import("vllm.models.minimax_m3.common.sparse_attention", _apply_patch)
