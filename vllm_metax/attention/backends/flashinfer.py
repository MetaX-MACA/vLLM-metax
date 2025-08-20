# SPDX-License-Identifier: Apache-2.0

import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple, Type

from flash_attn import flash_attn_varlen_func
import torch

from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import (AttentionLayer)
from vllm.logger import init_logger

logger = init_logger(__name__)

from vllm.attention.backends.flashinfer import (FlashInferBackend,
                                                FlashInferImpl,
                                                FlashInferMetadata)

class MetaxFlashInferBackend(FlashInferBackend):

    @staticmethod
    def get_impl_cls() -> Type["FlashInferImpl"]:
        return MetaxFlashInferImpl

class MetaxFlashInferImpl(FlashInferImpl):

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashInferMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # TODO: directly write to output tensor
        num_heads: int = self.num_heads
        head_size: int = self.head_size
        num_kv_heads: int = self.num_kv_heads
        kv_cache_dtype: str = self.kv_cache_dtype
        softmax_scale: float = self.scale
        window_size = self.sliding_window
        alibi_slopes = self.alibi_slopes
        logits_soft_cap = self.logits_soft_cap

        num_tokens, hidden_size = query.shape
        query = query.view(-1, num_heads, head_size)
        key = key.view(-1, num_kv_heads, head_size)
        value = value.view(-1, num_kv_heads, head_size)

        if kv_cache.numel() > 0:
            # Use the same reshape and cache kernel as flash attention.
            ops.reshape_and_cache_flash(
                key,
                value,
                kv_cache[:, 0],
                kv_cache[:, 1],
                attn_metadata.slot_mapping.flatten(),
                kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )
            # The FlashInfer api requires data to be in fp8_e4m3 or fp8_e5m2
            # to process the cache when the kv_cache_dtype is fp8
            if kv_cache_dtype.startswith("fp8"):
                torch_dtype = FlashInferBackend.get_fp8_dtype_for_flashinfer(
                    kv_cache_dtype)
                kv_cache = kv_cache.view(torch_dtype)

        num_prefill_tokens = attn_metadata.num_prefill_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        assert key.shape[0] == num_prefill_tokens + num_decode_tokens, \
                    f"key : {key.shape} : #prefill tokens {num_prefill_tokens} : #decode tokens {num_decode_tokens}" # noqa
        assert value.shape[0] == num_prefill_tokens + num_decode_tokens, \
                    f"value : {value.shape} : #prefill toks {num_prefill_tokens} : #decode toks {num_decode_tokens}" # noqa
        query = query.contiguous(
        )  # Flashinfer requires query to be contiguous
        # Query for decode. KV is not needed because it is already cached.
        # QKV for prefill.
        decode_query = query[num_prefill_tokens:]
        query = query[:num_prefill_tokens]

        key = key[:num_prefill_tokens]
        value = value[:num_prefill_tokens]

        assert query.shape[0] == num_prefill_tokens
        assert decode_query.shape[0] == num_decode_tokens

        window_left = window_size[0] if window_size is not None else -1

        prefill_output: Optional[torch.Tensor] = None
        decode_output: Optional[torch.Tensor] = None
        stride_order = FlashInferBackend.get_kv_cache_stride_order()
        if prefill_meta := attn_metadata.prefill_metadata:
            # We will use flash attention for prefill
            # when kv_cache is not provided.
            # This happens when vllm runs the profiling to
            # determine the number of blocks.
            if kv_cache.numel() == 0:
                prefill_output = flash_attn_varlen_func(
                    q=query,
                    k=key,
                    v=value,
                    cu_seqlens_q=prefill_meta.seq_start_loc,
                    cu_seqlens_k=prefill_meta.seq_start_loc,
                    max_seqlen_q=prefill_meta.max_prefill_seq_len,
                    max_seqlen_k=prefill_meta.max_prefill_seq_len,
                    softmax_scale=softmax_scale,
                    causal=True,
                    window_size=window_size,
                    alibi_slopes=alibi_slopes,
                )
            else:
                assert prefill_meta is not None
                assert prefill_meta.prefill_wrapper is not None

                assert prefill_meta.prefill_wrapper._causal
                assert prefill_meta.prefill_wrapper._window_left == window_left
                assert prefill_meta.prefill_wrapper._logits_soft_cap == (
                    logits_soft_cap or 0.0)
                assert prefill_meta.prefill_wrapper._sm_scale == softmax_scale

                prefill_output = prefill_meta.prefill_wrapper.run(
                    query,
                    kv_cache.permute(*stride_order),
                    k_scale=layer._k_scale_float,
                    v_scale=layer._v_scale_float,
                )
        if decode_meta := attn_metadata.decode_metadata:
            assert decode_meta is not None
            assert decode_meta.decode_wrapper is not None

            assert decode_meta.decode_wrapper._window_left == window_left
            assert decode_meta.decode_wrapper._logits_soft_cap == (
                logits_soft_cap or 0.0)
            assert decode_meta.decode_wrapper._sm_scale == softmax_scale

            decode_output = decode_meta.decode_wrapper.run(
                decode_query,
                kv_cache.permute(*stride_order),
                k_scale=layer._k_scale_float,
                v_scale=layer._v_scale_float,
            )

        if prefill_output is None and decode_output is not None:
            # Decode only batch.
            output, num_tokens = decode_output, num_decode_tokens
        elif decode_output is None and prefill_output is not None:
            # Prefill only batch.
            output, num_tokens = prefill_output, num_prefill_tokens
        else:
            # Chunked prefill batch does not work with speculative decoding in
            # FlashInfer backend, so the query length for decode should be 1.
            assert prefill_output is not None
            assert decode_output is not None
            assert decode_meta is not None
            assert decode_meta.decode_query_len == 1
            decode_output = decode_output.squeeze(1)
            output = torch.cat([prefill_output, decode_output], dim=0)
        return output.view(num_tokens, hidden_size)
