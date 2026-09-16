# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: `layer.topk_indices_buffer` (the shared, cudagraph-stable top-k
#       output buffer written in indexer.py and read back here) is allocated
#       TOKEN-MAJOR by nvidia/model.py: torch.empty(padded_num_tokens,
#       num_index_heads, topk_blocks, ...), i.e. [total_q, num_index_heads,
#       topk], with a comment promising "the attend transposes to
#       [H, tokens, topk]". MiniMaxM3SparseTritonImpl.forward never performs
#       that transpose; it slices the buffer assuming the OLD head-major
#       layout ([num_index_heads, total_q, topk], how amd/model.py still
#       allocates it, so ROCm is unaffected).
#
#       Whenever num_index_heads == 1 (true for MiniMax-M3 at any TP degree
#       where sparse_num_index_heads // tp_size rounds down to 0, e.g.
#       MiniMax-M3-W8A8 @ TP=8: max(1, 4 // 8) == 1), this hides almost
#       perfectly: decode's buf[:, :nd, :] and prefill's buf[:, 0:, :]
#       (nd == 0, i.e. every pure-prefill step, including a fresh engine's
#       very first step) both clamp to the whole size-1 dim and land on
#       offset 0 by accident. The first time a batch mixes decode with
#       prefill (nd > 0 -- from the 2nd engine step onward under any real
#       multi-request load), prefill's slice desyncs from the true token
#       offset, corrupting the topk indices the attention kernel uses to
#       index the paged KV-cache block_table -- surfacing as an illegal
#       memory access. Confirmed independently on MetaX C550; the same
#       crash is reported upstream on plain NVIDIA H800
#       (vllm-project/vllm#49147) -- this is a vLLM correctness bug, not a
#       MetaX- or Triton-backend issue. See indexer.py's matching
#       write-side patch in this same directory.
#
# Affected versions: v0.24.0 - v0.26.0 (checked the buffer-fetch line in
#       each tagged source directly).
#
# Remove at: vLLM dependency bumped to v0.27.0 or later (fixed upstream by
#       vllm-project/vllm#49149, commit d1a8ba63d9d2, merged 2026-07-25,
#       closing vllm-project/vllm#49147).
# -----------------------------------------------------------------------------
"""Backport of vllm-project/vllm#49149's read-side fix.

Copies MiniMaxM3SparseTritonImpl.forward verbatim from vLLM v0.26.0 (including
the k_scale/v_scale FP8 KV-cache scaling added after v0.24.0) with only the
topk buffer transpose inserted; see the module note above.
"""

from vllm.forward_context import get_forward_context
from vllm.models.minimax_m3.common.sparse_attention import (
    MiniMaxM3SparseMetadata,
    minimax_m3_sparse_attn,
    minimax_m3_sparse_attn_decode,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionLayer

from vllm_metax.patch.utils import patch


@patch(
    "vllm.models.minimax_m3.common.sparse_attention",
    "MiniMaxM3SparseTritonImpl.forward",
)
def forward(self, layer: AttentionLayer, query, kv_cache, output):
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

    # /-------------------- MetaX Modification --------------------\
    topk = (
        topk_buffer
        if current_platform.is_rocm()
        else topk_buffer[:num_tokens].transpose(0, 1)
    )
    # \-------------------- MetaX Modification --------------------/
    assert topk is not None
    hd = self.head_size
    q = query[:num_tokens].view(-1, self.num_heads, hd)
    out = output[:num_tokens].view(-1, self.num_heads, hd)
    kv_cache = kv_cache.view(self.kv_cache_fp8_dtype) if self.use_fp8_kv else kv_cache
    k_scale = getattr(layer, "_k_scale", None) if self.use_fp8_kv else None
    v_scale = getattr(layer, "_v_scale", None) if self.use_fp8_kv else None

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
            k_scale=k_scale,
            v_scale=v_scale,
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
            k_scale=k_scale,
            v_scale=v_scale,
        )
    return output
