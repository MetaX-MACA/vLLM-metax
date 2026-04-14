# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom Sparse Attention Indexer layers."""

import torch


from vllm.logger import init_logger

from vllm.model_executor.layers.sparse_attn_indexer import SparseAttnIndexer
from . import bf16, fp8  # noqa: F401

logger = init_logger(__name__)


@SparseAttnIndexer.register_oot
class MacaSparseAttnIndexer(SparseAttnIndexer):
    def __init__(
        self,
        k_cache,
        quant_block_size: int,
        scale_fmt: str,
        topk_tokens: int,
        head_dim: int,
        max_model_len: int,
        max_total_seq_len: int,
        topk_indices_buffer: torch.Tensor,
    ):
        super(SparseAttnIndexer, self).__init__()
        self.k_cache = k_cache
        self.quant_block_size = quant_block_size
        self.scale_fmt = scale_fmt
        self.topk_tokens = topk_tokens
        self.head_dim = head_dim
        self.max_model_len = max_model_len
        self.max_total_seq_len = max_total_seq_len
        self.topk_indices_buffer = topk_indices_buffer

    def forward_oot(
        self,
        hidden_states: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        if q.dtype in (torch.bfloat16, torch.float16):
            sparse_attn_indexer_impl = torch.ops.vllm.mx_sparse_attn_indexer_bf16
        else:
            sparse_attn_indexer_impl = torch.ops.vllm.mx_sparse_attn_indexer

        return sparse_attn_indexer_impl(
            hidden_states,
            self.k_cache.prefix,
            self.k_cache.kv_cache,
            q,
            k,
            weights,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
        )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        q_fp8: torch.Tensor,
        k: torch.Tensor,
        weights: torch.Tensor,
    ):
        return self.forward_oot(hidden_states, q_fp8, k, weights)
