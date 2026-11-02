# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch
from vllm_metax.utils.deep_gemm import (
    bf16_mqa_logits,
    bf16_paged_mqa_logits,
    fp8_mqa_logits,
    fp8_paged_mqa_logits,
    int8_mqa_logits,
    int8_paged_mqa_logits,
)
from .kpool_bf16 import (
    gather_paged_k_bf16,
    kpool_compress_insert_bf16,
    kpool_decode_update_and_maybe_write_cache_bf16,
)
from .kpool_int8 import (
    kpool_compress_insert_int8,
    kpool_decode_update_and_maybe_write_cache_int8,
)


def _mqa_logits_by_dtype(
    q: torch.Tensor,
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool,
) -> torch.Tensor:
    if q.dtype in (torch.bfloat16, torch.float16):
        return bf16_mqa_logits(q, kv[0], weights, cu_seqlen_ks, cu_seqlen_ke)
    if q.dtype == torch.int8:
        return int8_mqa_logits(
            q,
            kv,
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            clean_logits=clean_logits,
        )
    return fp8_mqa_logits(
        q,
        kv,
        weights,
        cu_seqlen_ks,
        cu_seqlen_ke,
        clean_logits=clean_logits,
    )


def _paged_mqa_logits_by_dtype(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
    clean_logits: bool,
) -> torch.Tensor:
    if q.dtype in (torch.bfloat16, torch.float16):
        return bf16_paged_mqa_logits(
            q,
            kv_cache,
            weights,
            context_lens,
            block_tables,
            schedule_metadata,
            max_model_len,
            clean_logits=clean_logits,
        )
    if q.dtype == torch.int8:
        return int8_paged_mqa_logits(
            q,
            kv_cache,
            weights,
            context_lens,
            block_tables,
            schedule_metadata,
            max_model_len,
            clean_logits=clean_logits,
        )
    return fp8_paged_mqa_logits(
        q,
        kv_cache,
        weights,
        context_lens,
        block_tables,
        schedule_metadata,
        max_model_len,
        clean_logits=clean_logits,
    )
