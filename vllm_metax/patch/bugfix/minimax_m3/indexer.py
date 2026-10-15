# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: The MiniMax-M3 index-score fix in `index_topk.py` replaces
#       `vllm.models.minimax_m3.common.ops.index_topk.minimax_m3_index_score`,
#       but upstream `common/indexer.py` imports the wrappers with a top-level
#       value import:
#
#           from vllm.models.minimax_m3.common.ops.index_topk import (
#               minimax_m3_index_decode,
#               minimax_m3_index_score,
#               minimax_m3_index_topk,
#           )
#
#       Importing `vllm.models.minimax_m3` (forced while the `index_topk.py`
#       patch module itself loads, via the package __init__ -> nvidia/model.py
#       cascade) executes that value import BEFORE the sibling patch installs
#       its replacements, so replacing the ops.index_topk module attribute
#       alone never reaches the live prefill call in
#       `MiniMaxM3IndexerTritonImpl.forward`. `forward` must therefore be
#       replaced too, looking the wrappers up on the ops module at call time.
#
#       The forward body below is upstream v0.27.0's verbatim (including the
#       token-major topk buffer transpose added by vllm-project/vllm#49149);
#       only the wrapper calls are re-pointed at the ops module.
#
# Affected versions: v0.27.0 (same caveat applies to v0.24.0-v0.26.0; the
#       vllm_metax v0.26.0 patch set carries this sibling indexer.py fix).
#
# Remove at: upstream changes common/indexer.py to import the wrappers after
#       plugin patches are installed, or to call through the ops module.
# -----------------------------------------------------------------------------
"""Rebind the MiniMax-M3 Triton indexer onto the patched index-topk wrappers.

The attribute-level patches in the sibling ``index_topk.py`` only take effect
for callers that look the wrappers up on the ops module at call time. Upstream
``common/indexer.py`` instead binds them through a top-level value import that
runs before those patches install, so this module replaces the Triton impl's
``forward`` with the upstream body re-pointed at the ops module.
"""

import torch

from vllm.forward_context import get_forward_context
from vllm.models.minimax_m3.common.indexer import MiniMaxM3IndexerMetadata
from vllm.platforms import current_platform

# Mirror upstream common/indexer.py's platform split, but resolve the module at
# call time inside forward so the patched wrappers are always picked up.
if current_platform.is_rocm():
    from vllm.models.minimax_m3.amd.ops import index_topk as _index_topk_ops
else:
    from vllm.models.minimax_m3.common.ops import index_topk as _index_topk_ops

from vllm_metax.patch.utils import patch


@patch("vllm.models.minimax_m3.common.indexer", "MiniMaxM3IndexerTritonImpl.forward")
def forward(
    self,
    index_query: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
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
    buf_htk = buf if buf is None or current_platform.is_rocm() else buf.transpose(0, 1)
    decode_topk: torch.Tensor | None = None
    prefill_topk: torch.Tensor | None = None
    if index_md.num_decodes > 0:
        d = index_md.decode
        assert d is not None
        decode_topk = _index_topk_ops.minimax_m3_index_decode(
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
        score = _index_topk_ops.minimax_m3_index_score(
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
        prefill_topk = _index_topk_ops.minimax_m3_index_topk(
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
