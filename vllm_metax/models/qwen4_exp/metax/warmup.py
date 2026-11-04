# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Compile MetaX QSA specializations through the runner's JIT warmup registry."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.model_executor.warmup.jit_warmup import VllmJitKernel

if TYPE_CHECKING:
    from .qsa import Qwen4ExpQSAAttention


class QSASparseAttentionWarmup(VllmJitKernel["QSASparseAttentionWarmup.CompileKey"]):
    @dataclass(frozen=True)
    class CompileKey:
        cache_shape: tuple[int, ...]
        cache_stride: tuple[int, ...]
        cache_dtype: torch.dtype
        block_table_width: int
        num_query_heads: int
        selection_width: int

    def get_warmup_keys(self, layer: "Qwen4ExpQSAAttention") -> list[CompileKey]:
        # NOTE(metax): Registration happens during model construction; the real
        # cache layout and metadata become available before registry.warmup().
        cache = layer.kv_cache
        assert cache.numel(), "MetaX QSA KV cache must be bound before warmup"
        widths = layer._qsa_warmup_block_table_widths
        assert widths, "MetaX QSA metadata must be initialized before warmup"
        return [
            self.CompileKey(
                cache_shape=tuple(cache.shape),
                cache_stride=tuple(cache.stride()),
                cache_dtype=cache.dtype,
                block_table_width=width,
                num_query_heads=layer.num_heads,
                selection_width=layer.indexer.output_width,
            )
            for width in widths
        ]

    def compile(self, compile_key: CompileKey) -> None:
        from .ops.qsa import warmup_qsa_sparse_paged_attention

        # NOTE(metax): Preserve the bound allocation's physical strides without
        # allocating another KV cache during warmup.
        cache = torch.empty_strided(
            compile_key.cache_shape,
            compile_key.cache_stride,
            dtype=compile_key.cache_dtype,
            device="meta",
        )
        block_table = torch.empty(
            (1, compile_key.block_table_width), dtype=torch.int32, device="meta"
        )
        warmup_qsa_sparse_paged_attention(
            cache,
            block_table,
            num_query_heads=compile_key.num_query_heads,
            selection_width=compile_key.selection_width,
        )


qsa_sparse_attention_warmup = QSASparseAttentionWarmup()
