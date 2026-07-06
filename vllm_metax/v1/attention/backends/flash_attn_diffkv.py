# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with FlashAttention."""

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import (
    canonicalize_singleton_dim_strides,
    is_quantized_kv_cache,
)
from vllm.v1.attention.backend import AttentionType
from vllm_metax.v1.attention.backends.fa_utils import (
    get_flash_attn_version,
    is_flash_attn_varlen_func_available,
)
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash_diffkv,
)

if is_flash_attn_varlen_func_available():
    from vllm_metax.v1.attention.backends.fa_utils import (
        flash_attn_varlen_func,
        flash_attn_with_kvcache,
    )
from vllm.v1.attention.backends.utils import (
    get_kv_cache_layout,
    reshape_attn_output_for_spec_decode,  # used for prefill decode split with mtp
    reshape_query_for_spec_decode,  # used for prefill decode split with mtp
)

from .flash_attn import (
    MacaFlashAttentionBackend as FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    cascade_attention,
)

import vllm_metax.envs as mx_envs

logger = init_logger(__name__)


class FlashAttentionDiffKVBackend(FlashAttentionBackend):
    # Default to 128 for this backend
    head_size_v: int = 128

    @classmethod
    def set_head_size_v(cls, head_size_v: int) -> None:
        cls.head_size_v = head_size_v

    @classmethod
    def is_supported_on_current_device(
        cls,
        head_size: int,
        head_size_v: int,
        has_sinks: bool,
    ) -> bool:
        return True

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_DIFFKV"

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        return FlashAttentionDiffKVImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (
            num_blocks,
            block_size,
            num_kv_heads,
            head_size + FlashAttentionDiffKVBackend.head_size_v,
        )

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # `stride_order` indicates the permutation that gets
        # us from `get_kv_cache_shape` to the actual memory layout we want.
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            # (num_blocks, num_layers, block_size,
            # num_kv_heads, head_size + head_size_v)
            return (1, 0, 2, 3, 4)
        elif cache_layout == "NHD":
            stride_order = (0, 1, 2, 3)
        elif cache_layout == "HND" and include_num_layers_dimension:
            # (num_blocks, num_kv_heads, num_layers,
            # block_size, head_size + head_size_v)
            return (1, 3, 0, 2, 4)
        elif cache_layout == "HND":
            stride_order = (0, 2, 1, 3)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")
        return stride_order


class FlashAttentionDiffKVImpl(FlashAttentionImpl):
    vllm_flash_attn_version: int | None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Re-derive the FA version with diff-kv context so that
        # get_flash_attn_version can apply the FA3 -> FA4 upgrade rule
        # for sinks + hdim != hdim_v.
        self.vllm_flash_attn_version = get_flash_attn_version(
            requires_alibi=self.alibi_slopes is not None,
            head_size=self.head_size,
            head_size_v=FlashAttentionDiffKVBackend.head_size_v,
            has_sinks=self.sinks is not None,
        )

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return

        # Unlike standard FlashAttn which splits kv_cache via unbind(0),
        # DiffKV packs K and V into a single tensor along the last dim:
        #   kv_cache shape: [num_blocks, block_size, num_kv_heads,
        #                    head_size_k + head_size_v]
        # The triton kernel handles this combined layout directly.
        #
        # NOTE(woosuk): key and value are padded while slot_mapping is
        # not padded. However, we don't need to do key[:num_actual_tokens]
        # and value[:num_actual_tokens] because the reshape_and_cache_flash
        # op uses the slot_mapping's shape to determine the number of
        # actual tokens.
        triton_reshape_and_cache_flash_diffkv(
            key,
            value,
            kv_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size_v]
            kv_cache: shape =
                [num_blocks, block_size, num_kv_heads, head_size + head_size_v]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size_v]
        NOTE: FP8 quantization, flash-attn expect the size of
              {q,k,v}_descale to be (num_sequences, num_kv_heads).
              We use torch's .expand() to avoid duplicating values
        """
        assert self.vllm_flash_attn_version is not None, (
            "FlashAttention version not detected."
        )

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for FlashAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        attn_type = self.attn_type

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # Handle encoder attention differently - no KV cache needed
        if attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return self._forward_encoder_attention(
                query[:num_actual_tokens],
                key[:num_actual_tokens],
                value[:num_actual_tokens],
                output[:num_actual_tokens],
                attn_metadata,
                layer,
            )

        # For decoder and cross-attention, use KV cache as before
        # Different head_size for K and V
        key_cache = kv_cache[..., : self.head_size]
        value_cache = kv_cache[..., self.head_size :]
        # Fix degenerate strides on size-1 dims (e.g. num_kv_heads=1 with TP).
        # FA3/4 on H100+ uses TMA, which requires ≥16-byte stride alignment.
        # See vllm.utils.torch_utils.canonicalize_singleton_dim_strides.
        fixed_k = canonicalize_singleton_dim_strides(key_cache)
        fixed_v = canonicalize_singleton_dim_strides(value_cache)
        if fixed_k is not key_cache or fixed_v is not value_cache:
            logger.debug(
                "Canonicalized degenerate KV cache strides (FlashAttentionDiffKV): "
                "shape=%s, key strides before=%s after=%s, "
                "value strides before=%s after=%s",
                key_cache.shape,
                key_cache.stride(),
                fixed_k.stride(),
                value_cache.stride(),
                fixed_v.stride(),
            )
        key_cache, value_cache = fixed_k, fixed_v

        if is_quantized_kv_cache(self.kv_cache_dtype):
            # queries are quantized in the attention layer
            key_cache = key_cache.view(current_platform.fp8_dtype())
            value_cache = value_cache.view(current_platform.fp8_dtype())

        if not attn_metadata.use_cascade:
            cu_seqlens_q = attn_metadata.query_start_loc
            seqused_k = attn_metadata.seq_lens
            max_seqlen_q = attn_metadata.max_query_len
            max_seqlen_k = attn_metadata.max_seq_len
            block_table = attn_metadata.block_table
            scheduler_metadata = attn_metadata.scheduler_metadata

            descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)

            if self.dcp_world_size > 1:
                self._forward_with_dcp(
                    query[:num_actual_tokens],
                    key[:num_actual_tokens],
                    value[:num_actual_tokens],
                    key_cache,
                    value_cache,
                    output[:num_actual_tokens],
                    attn_metadata,
                    q_descale=layer._q_scale.expand(descale_shape),
                    k_descale=layer._k_scale.expand(descale_shape),
                    v_descale=layer._v_scale.expand(descale_shape),
                )
                return output
            else:
                sliding_window_size = (
                    list(self.sliding_window)
                    if self.sliding_window is not None
                    else None
                )
                if mx_envs.VLLM_METAX_ENABLE_FA_SPLIT_FORWARD:
                    # ┌------------------------  Metax Modification -------------------------┐
                    # For handling prefill decode split
                    num_decode_tokens = attn_metadata.num_decode_tokens
                    if attn_metadata.num_prefills > 0:
                        output[num_decode_tokens:num_actual_tokens] = (
                            flash_attn_varlen_func(
                                q=query[num_decode_tokens:num_actual_tokens],
                                k=key_cache,
                                v=value_cache,
                                cu_seqlens_q=attn_metadata.prefill_query_start_loc,
                                cu_seqlens_k=attn_metadata.cu_prefix_kv_lens,
                                max_seqlen_q=attn_metadata.max_query_len,
                                max_seqlen_k=attn_metadata.prefill_max_seq_len,
                                softmax_scale=self.scale,
                                causal=attn_metadata.causal,
                                alibi_slopes=self.alibi_slopes,
                                window_size=sliding_window_size,
                                block_table=attn_metadata.prefill_block_table,
                                softcap=self.logits_soft_cap,
                                s_aux=self.sinks,
                            )
                        )
                    if attn_metadata.num_decodes > 0:
                        decode_query = query[:num_decode_tokens]
                        # Use flash_attn_with_kvcache for normal decoding.
                        if attn_metadata.decode_bucket_req_bounds is not None:
                            self._forward_decode_with_query_len_bucketing(
                                decode_query,
                                key_cache,
                                value_cache,
                                output,
                                attn_metadata,
                            )
                        else:
                            decode_query = reshape_query_for_spec_decode(
                                decode_query, attn_metadata.num_decodes
                            )
                            output_unreshape = flash_attn_with_kvcache(
                                q=decode_query,
                                k_cache=key_cache,
                                v_cache=value_cache,
                                block_table=attn_metadata.decode_block_table,
                                cache_seqlens=attn_metadata.decode_seq_lens,
                                softmax_scale=self.scale,
                                causal=True,
                                window_size=sliding_window_size,
                                alibi_slopes=self.alibi_slopes,
                                softcap=self.logits_soft_cap,
                                s_aux=self.sinks,
                            )
                            output[:num_decode_tokens] = (
                                reshape_attn_output_for_spec_decode(output_unreshape)
                            )
                    return output
                # └------------------------- Metax Modification -------------------------┘
                else:
                    # cu_seqlens_k = attn_metadata.cu_seqlens_k
                    # if cu_seqlens_k is None:
                    # Fallback for legacy metadata paths: keep it GPU-only.
                    # TODO(hank): Currently we manually process it on forward. Move it to attention_metadata
                    cu_seqlens_k = F.pad(
                        attn_metadata.seq_lens,
                        (1, 0),
                        value=0,
                    ).cumsum(dim=0, dtype=torch.int32)

                    output[:num_actual_tokens] = flash_attn_varlen_func(
                        q=query[:num_actual_tokens],
                        k=key_cache,
                        v=value_cache,
                        cu_seqlens_q=cu_seqlens_q,
                        max_seqlen_q=max_seqlen_q,
                        cu_seqlens_k=cu_seqlens_k,
                        max_seqlen_k=max_seqlen_k,
                        softmax_scale=self.scale,
                        causal=True,
                        alibi_slopes=self.alibi_slopes,
                        window_size=sliding_window_size,
                        block_table=block_table,
                        softcap=self.logits_soft_cap,
                    )
                    return output

        # Cascade attention (rare case).
        cascade_attention(
            output[:num_actual_tokens],
            query[:num_actual_tokens],
            key_cache,
            value_cache,
            cu_query_lens=attn_metadata.query_start_loc,
            max_query_len=attn_metadata.max_query_len,
            cu_prefix_query_lens=attn_metadata.cu_prefix_query_lens,
            prefix_kv_lens=attn_metadata.prefix_kv_lens,
            suffix_kv_lens=attn_metadata.suffix_kv_lens,
            max_kv_len=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window,
            logits_soft_cap=self.logits_soft_cap,
            block_table=attn_metadata.block_table,
            common_prefix_len=attn_metadata.common_prefix_len,
            max_num_splits=attn_metadata.max_num_splits,
            fa_version=self.vllm_flash_attn_version,
            prefix_scheduler_metadata=attn_metadata.prefix_scheduler_metadata,
            suffix_scheduler_metadata=attn_metadata.scheduler_metadata,
            q_descale=layer._q_scale,
            k_descale=layer._k_scale,
            v_descale=layer._v_scale,
            s_aux=self.sinks,
        )
        return output
