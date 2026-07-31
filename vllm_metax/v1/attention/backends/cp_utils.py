# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import torch

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed import get_dcp_group
from vllm.logger import init_logger
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.utils import split_decodes_prefills_and_extends

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
else:
    AttentionLayerBase = object


def run_split_fa2_dcp_context_attention(
    flash_attn_varlen_func: Any,
    query_across_dcp: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    dcp_context_out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    dcp_context_kv_lens: torch.Tensor,
    max_dcp_context_kv_len: int,
    softmax_scale: float,
    alibi_slopes: torch.Tensor | None,
    sliding_window_size: list[int] | None,
    block_table: torch.Tensor,
    softcap: float,
    fa_version: int,
    q_descale: torch.Tensor | None,
    k_descale: torch.Tensor | None,
    v_descale: torch.Tensor | None,
    max_num_splits: int,
    num_heads: int,
    dcp_world_size: int,
    num_decode_reqs: int,
    num_context_prefill_reqs: int,
    num_decode_tokens: int,
    num_context_prefill_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    dcp_context_out.zero_()
    context_lse = torch.full(
        (num_heads * dcp_world_size, query_across_dcp.shape[0]),
        -torch.inf,
        dtype=torch.float32,
        device=query_across_dcp.device,
    )

    if num_decode_tokens > 0:
        dcp_context_out[:num_decode_tokens], decode_context_lse = (
            flash_attn_varlen_func(
                q=query_across_dcp[:num_decode_tokens],
                k=key_cache,
                v=value_cache,
                cu_seqlens_q=cu_seqlens_q[: num_decode_reqs + 1],
                max_seqlen_q=1,
                seqused_k=dcp_context_kv_lens[:num_decode_reqs],
                max_seqlen_k=max_dcp_context_kv_len,
                softmax_scale=softmax_scale,
                causal=False,
                alibi_slopes=alibi_slopes,
                window_size=sliding_window_size,
                block_table=block_table[:num_decode_reqs],
                softcap=softcap,
                return_softmax_lse=True,
            )
        )
        context_lse[:, :num_decode_tokens] = decode_context_lse

    if num_context_prefill_tokens > 0:
        prefill_start = num_decode_tokens
        prefill_end = prefill_start + num_context_prefill_tokens
        prefill_query_start_loc = (
            cu_seqlens_q[
                num_decode_reqs : num_decode_reqs + num_context_prefill_reqs + 1
            ]
            - num_decode_tokens
        )
        prefill_req_slice = slice(
            num_decode_reqs, num_decode_reqs + num_context_prefill_reqs
        )
        dcp_context_out[prefill_start:prefill_end], prefill_context_lse = (
            flash_attn_varlen_func(
                q=query_across_dcp[prefill_start:prefill_end],
                k=key_cache,
                v=value_cache,
                cu_seqlens_q=prefill_query_start_loc,
                max_seqlen_q=max_seqlen_q,
                seqused_k=dcp_context_kv_lens[prefill_req_slice],
                max_seqlen_k=max_dcp_context_kv_len,
                softmax_scale=softmax_scale,
                causal=False,
                alibi_slopes=alibi_slopes,
                window_size=sliding_window_size,
                block_table=block_table[prefill_req_slice],
                softcap=softcap,
                return_softmax_lse=True,
            )
        )
        context_lse[:, prefill_start:prefill_end] = prefill_context_lse

    return dcp_context_out, context_lse
