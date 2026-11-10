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
    is_fa_version_supported,
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
    reshape_attn_output_for_spec_decode,  # used for prefill decode split with mtp
    reshape_query_for_spec_decode,  # used for prefill decode split with mtp
)

from .flash_attn import (
    MacaFlashAttentionBackend as FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

import vllm_metax.envs as mx_envs

logger = init_logger(__name__)


@register_backend(AttentionBackendEnum.FLASH_ATTN_DIFFKV)
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
        """MetaX FA2/FA3 support the exact QK/V pair 192/128."""
        if (head_size, head_size_v) != (192, 128):
            return False
        if has_sinks:
            return is_fa_version_supported(2)
        return is_flash_attn_varlen_func_available()

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN_DIFFKV"

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        return FlashAttentionDiffKVImpl


class FlashAttentionDiffKVImpl(FlashAttentionImpl):
    vllm_flash_attn_version: int | None

    def __init__(self, *args, **kwargs) -> None:
        self.head_size_v = FlashAttentionDiffKVBackend.head_size_v
        super().__init__(*args, **kwargs)
        self.fa4_hd256 = False

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

        # DiffKV packs K and V into a single tensor along the last dim:
        #   kv_cache shape: [num_blocks, block_size, num_kv_heads,
        #                    head_size_k + head_size_v]
        # (B, H, N, C) -> (B, N, H, C) for kernel compatibility.
        triton_reshape_and_cache_flash_diffkv(
            key,
            value,
            kv_cache.transpose(1, 2),
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
            layer: The attention layer, providing the q/k/v quantization scales.
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size_v]
            kv_cache: shape =
                [num_blocks, block_size, num_kv_heads, head_size + head_size_v]
            attn_metadata: Metadata for attention.
            output: Tensor that the attention result is written into.
            output_scale: Scale for fused output quantization; not supported
                by this backend.
            output_block_scale: Block scale for fused output quantization;
                not supported by this backend.

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

        # (B, H, N, 2*hs) -> ((B, N, H, hs), (B, N, H, hs))
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
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

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens  # noqa: F841
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len
        block_table = attn_metadata.block_table
        scheduler_metadata = attn_metadata.scheduler_metadata  # noqa: F841

        descale_shape = (cu_seqlens_q.shape[0] - 1, self.num_kv_heads)
        q_descale = (
            layer._q_scale.expand(descale_shape)
            if self.vllm_flash_attn_version == 3
            else None
        )
        k_descale = (
            layer._k_scale.expand(descale_shape)
            if self.vllm_flash_attn_version == 3
            else None
        )
        v_descale = (
            layer._v_scale.expand(descale_shape)
            if self.vllm_flash_attn_version == 3
            else None
        )

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
                list(self.sliding_window) if self.sliding_window is not None else None
            )
            if mx_envs.VLLM_METAX_ENABLE_FA_SPLIT_FORWARD:
                # ┌------------------------  Metax Modification -------------------------┐
                # For handling prefill decode split
                split_decode_num_tokens = attn_metadata.split_decode_num_tokens
                if attn_metadata.split_prefill_num_reqs > 0:
                    if self.vllm_flash_attn_version == 3:
                        # NOTE(metax): FA3 varlen cannot read paged KV, including cached prefixes.
                        num_decodes = attn_metadata.split_decode_num_reqs
                        flash_attn_with_kvcache(
                            q=query[split_decode_num_tokens:num_actual_tokens],
                            k_cache=key_cache,
                            v_cache=value_cache,
                            out=output[split_decode_num_tokens:num_actual_tokens],
                            cu_seqlens_q=attn_metadata.split_prefill_query_start_loc,
                            max_seqlen_q=max_seqlen_q,
                            cache_seqlens=attn_metadata.seq_lens[num_decodes:],
                            block_table=attn_metadata.split_prefill_block_table,
                            softmax_scale=self.scale,
                            causal=attn_metadata.causal,
                            window_size=sliding_window_size,
                            softcap=self.logits_soft_cap,
                            q_descale=q_descale[num_decodes:]
                            if q_descale is not None
                            else None,
                            k_descale=k_descale[num_decodes:]
                            if k_descale is not None
                            else None,
                            v_descale=v_descale[num_decodes:]
                            if v_descale is not None
                            else None,
                            num_splits=1 if self.batch_invariant_enabled else 0,
                            fa_version=self.vllm_flash_attn_version,
                        )
                    else:
                        output[split_decode_num_tokens:num_actual_tokens] = (
                            flash_attn_varlen_func(
                                q=query[split_decode_num_tokens:num_actual_tokens],
                                k=key_cache,
                                v=value_cache,
                                cu_seqlens_q=(
                                    attn_metadata.split_prefill_query_start_loc
                                ),
                                cu_seqlens_k=attn_metadata.split_prefill_cu_seqlens_k,
                                max_seqlen_q=attn_metadata.max_query_len,
                                max_seqlen_k=attn_metadata.split_prefill_max_seq_len,
                                softmax_scale=self.scale,
                                causal=attn_metadata.causal,
                                alibi_slopes=self.alibi_slopes,
                                window_size=sliding_window_size,
                                block_table=attn_metadata.split_prefill_block_table,
                                softcap=self.logits_soft_cap,
                                s_aux=self.sinks,
                                fa_version=self.vllm_flash_attn_version,
                            )
                        )
                if attn_metadata.split_decode_num_reqs > 0:
                    decode_query = query[:split_decode_num_tokens]
                    num_decodes = attn_metadata.split_decode_num_reqs
                    decode_q_descale = (
                        q_descale[:num_decodes] if q_descale is not None else None
                    )
                    decode_k_descale = (
                        k_descale[:num_decodes]
                        if self.vllm_flash_attn_version == 3 and k_descale is not None
                        else None
                    )
                    decode_v_descale = (
                        v_descale[:num_decodes]
                        if self.vllm_flash_attn_version == 3 and v_descale is not None
                        else None
                    )
                    # Use flash_attn_with_kvcache for normal decoding.
                    if attn_metadata.split_decode_bucket_req_bounds is not None:
                        self._forward_decode_with_query_len_bucketing(
                            decode_query,
                            key_cache,
                            value_cache,
                            output,
                            attn_metadata,
                            q_descale=decode_q_descale,
                            k_descale=decode_k_descale,
                            v_descale=decode_v_descale,
                        )
                    else:
                        decode_query = reshape_query_for_spec_decode(
                            decode_query,
                            attn_metadata.split_decode_num_reqs,
                        )
                        output_unreshape = flash_attn_with_kvcache(
                            q=decode_query,
                            k_cache=key_cache,
                            v_cache=value_cache,
                            block_table=attn_metadata.split_decode_block_table,
                            cache_seqlens=attn_metadata.split_decode_seq_lens,
                            softmax_scale=self.scale,
                            causal=attn_metadata.causal,
                            window_size=sliding_window_size,
                            alibi_slopes=self.alibi_slopes,
                            softcap=self.logits_soft_cap,
                            s_aux=self.sinks,
                            num_splits=1 if self.batch_invariant_enabled else 0,
                            fa_version=self.vllm_flash_attn_version,
                            q_descale=decode_q_descale,
                            k_descale=decode_k_descale,
                            v_descale=decode_v_descale,
                        )
                        output[:split_decode_num_tokens] = (
                            reshape_attn_output_for_spec_decode(output_unreshape)
                        )
                return output
            # └------------------------- Metax Modification -------------------------┘
            else:
                if self.vllm_flash_attn_version == 3:
                    flash_attn_with_kvcache(
                        q=query[:num_actual_tokens],
                        k_cache=key_cache,
                        v_cache=value_cache,
                        out=output[:num_actual_tokens],
                        cu_seqlens_q=cu_seqlens_q,
                        max_seqlen_q=max_seqlen_q,
                        cache_seqlens=attn_metadata.seq_lens,
                        block_table=block_table,
                        softmax_scale=self.scale,
                        causal=attn_metadata.causal,
                        window_size=sliding_window_size,
                        softcap=self.logits_soft_cap,
                        q_descale=q_descale,
                        k_descale=k_descale,
                        v_descale=v_descale,
                        num_splits=1 if self.batch_invariant_enabled else 0,
                        fa_version=self.vllm_flash_attn_version,
                    )
                else:
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
                        causal=attn_metadata.causal,
                        alibi_slopes=self.alibi_slopes,
                        window_size=sliding_window_size,
                        block_table=block_table,
                        softcap=self.logits_soft_cap,
                        s_aux=self.sinks,
                        fa_version=self.vllm_flash_attn_version,
                    )
                return output
