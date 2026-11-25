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
from vllm_metax.models.glm5next.metax.kpool_bf16 import (
    gather_paged_k_bf16,
    kpool_compress_insert_bf16,
    kpool_decode_update_and_maybe_write_cache_bf16,
)
from vllm_metax.models.glm5next.metax.kpool_int8 import (
    kpool_compress_insert_int8,
    kpool_decode_update_and_maybe_write_cache_int8,
)


def _mqa_logits_by_dtype(
    q: torch.Tensor | tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool,
    logits_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if isinstance(q, tuple):
        q, q_scale = q
        if q_scale is not None:
            raise NotImplementedError(
                "Scaled MXFP4 queries require FP8/FP4 logits dispatch"
            )
    if q.dtype in (torch.bfloat16, torch.float16):
        logits = bf16_mqa_logits(q, kv[0], weights, cu_seqlen_ks, cu_seqlen_ke)
    elif q.dtype == torch.int8:
        logits = int8_mqa_logits(
            q,
            kv,
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            clean_logits=clean_logits,
        )
    else:
        logits = fp8_mqa_logits(
            q,
            kv,
            weights,
            cu_seqlen_ks,
            cu_seqlen_ke,
            clean_logits=clean_logits,
        )
    # The local top-k operators consume FP32, even for BF16 logits.

    return logits.to(logits_dtype).float()


def _paged_mqa_logits_by_dtype(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
    clean_logits: bool,
    logits_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if q.dtype in (torch.bfloat16, torch.float16):
        logits = bf16_paged_mqa_logits(
            q,
            kv_cache,
            weights,
            context_lens,
            block_tables,
            schedule_metadata,
            max_model_len,
            clean_logits=clean_logits,
        )
    elif q.dtype == torch.int8:
        logits = int8_paged_mqa_logits(
            q,
            kv_cache,
            weights,
            context_lens,
            block_tables,
            schedule_metadata,
            max_model_len,
            clean_logits=clean_logits,
        )
    else:
        logits = fp8_paged_mqa_logits(
            q,
            kv_cache,
            weights,
            context_lens,
            block_tables,
            schedule_metadata,
            max_model_len,
            clean_logits=clean_logits,
        )
    return logits.to(logits_dtype).float()
