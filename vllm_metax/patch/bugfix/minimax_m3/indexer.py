# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: `self.topk_indices_buffer` (the shared, cudagraph-stable top-k
#       output buffer written here and read back in sparse_attention.py) is
#       allocated TOKEN-MAJOR by nvidia/model.py: torch.empty(
#       padded_num_tokens, num_index_heads, topk_blocks, ...), i.e.
#       [total_q, num_index_heads, topk], with a comment promising "the
#       attend transposes to [H, tokens, topk]". This impl's out= slicing
#       below assumes the OLD head-major layout ([num_index_heads, total_q,
#       topk], how amd/model.py still allocates it, so ROCm is unaffected)
#       and never performs that promised transpose.
#
#       Whenever num_index_heads == 1 (true for MiniMax-M3 at any TP degree
#       where sparse_num_index_heads // tp_size rounds down to 0, e.g.
#       MiniMax-M3-W8A8 @ TP=8: max(1, 4 // 8) == 1), this hides almost
#       perfectly until the first batch that mixes decode with prefill
#       (nd > 0). See sparse_attention.py's matching read-side patch in this
#       same directory for the full write-up, including the link to
#       vllm-project/vllm#49147 / #49149.
#
# Affected versions: v0.24.0 - v0.26.0 (checked the buffer-fetch line in
#       each tagged source directly).
#
# Remove at: vLLM dependency bumped to v0.27.0 or later (fixed upstream by
#       vllm-project/vllm#49149, commit d1a8ba63d9d2, merged 2026-07-25,
#       closing vllm-project/vllm#49147).
# -----------------------------------------------------------------------------
"""Backport of vllm-project/vllm#49149's write-side fix.

Copies MiniMaxM3IndexerTritonImpl.forward verbatim from vLLM v0.26.0
(unchanged since v0.24.0) with only the topk buffer transpose inserted; see
the module note above.
"""

import torch

from vllm.forward_context import get_forward_context
from vllm.models.minimax_m3.common.indexer import MiniMaxM3IndexerMetadata

# Import the kernel wrappers from their defining module (ops.index_topk),
# NOT from common.indexer's own `from ...ops.index_topk import ...` -- that
# is a value-import that copies whatever ops.index_topk.minimax_m3_index_*
# was bound to at the time common.indexer first loaded (forced early by
# vllm.models.minimax_m3's package __init__ cascade, before this patch
# module's own bugfix.minimax_m3.index_topk sibling has had a chance to
# install its fix there). Sourcing directly from ops.index_topk always
# reflects the currently-installed (patched) version, independent of import
# order. See bugfix/minimax_m3/index_topk.py for that fix.
from vllm.models.minimax_m3.common.ops.index_topk import (
    minimax_m3_index_decode,
    minimax_m3_index_score,
    minimax_m3_index_topk,
)
from vllm.platforms import current_platform

from vllm_metax.patch.utils import patch


@patch("vllm.models.minimax_m3.common.indexer", "MiniMaxM3IndexerTritonImpl.forward")
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
    # /-------------------- MetaX Modification --------------------\
    buf_htk = buf if buf is None or current_platform.is_rocm() else buf.transpose(0, 1)
    # \-------------------- MetaX Modification --------------------/
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
