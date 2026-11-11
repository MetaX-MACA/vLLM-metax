# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any, cast

import torch
from torch import nn

from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm_metax.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    compress_norm_rope_store_two_stage_triton,
)
from vllm_metax.models.deepseek_v4.common.ops.fused_indexer_q import MXFP4_BLOCK_SIZE
from vllm_metax.models.deepseek_v4.common.ops.fused_compress_quant_cache_metax import (
    _FUSED_KV_COMPRESS_NORM_ROPE_INSERT_INDEXER_TRITON_KERNEL,
    compress_norm_rope_store_triton,
)
from vllm_metax.models.deepseek_v4.common.ops.save_partial_states import (
    _SAVE_PARTIAL_STATES_KERNEL,
)
from vllm.platforms import current_platform

from vllm_metax.models.deepseek_v4.metax.compressor_base import (
    CompressorMetadata,
    CompressorStateCache,
)


class MacaDeepseekCompressor(nn.Module):
    """DeepSeek V4 KV/score compressor with MetaX cache formats and warmup."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        compress_ratio: int,
        hidden_size: int,
        head_dim: int,
        rotate: bool = False,
        prefix: str = "",
        k_cache_prefix="",
        use_fp4_cache: bool = False,
    ):
        super().__init__()
        self.compress_ratio = compress_ratio
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.rotate = rotate
        self.prefix = prefix
        self.k_cache_prefix = k_cache_prefix
        self.use_fp4_cache = use_fp4_cache

        config = vllm_config.model_config.hf_config
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.rms_norm_eps = config.rms_norm_eps
        self.device = current_platform.device_type
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len = vllm_config.model_config.max_model_len

        self.overlap = compress_ratio == 4
        self.coff = 1 + self.overlap

        # Upstream's two-stage path is ROCm-only; MACA uses the fused Triton
        # path for both C4 and C128 compressors.
        self._use_two_stage_fused_compressor = (
            not current_platform.is_maca() and head_dim == 512 and not self.overlap
        )
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )
        self._compress_scratch: torch.Tensor | None = None
        if self._use_two_stage_fused_compressor:
            self._compress_scratch = torch.empty(
                self.max_num_batched_tokens,
                self.head_dim,
                dtype=torch.float32,
                device=self.device,
            )

        state_dtype = torch.float32
        self.ape = nn.Parameter(
            torch.empty(
                (compress_ratio, self.coff * self.head_dim),
                dtype=state_dtype,
                device=self.device,
            ),
            requires_grad=False,
        )

        self.fused_wkv_wgate = MergedColumnParallelLinear(
            self.hidden_size,
            [self.coff * self.head_dim, self.coff * self.head_dim],
            bias=False,
            return_bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.fused_wkv_wgate",
        )
        self.norm = RMSNorm(self.head_dim, self.rms_norm_eps)

        self.state_cache = CompressorStateCache(
            state_dim=2 * self.coff * self.head_dim,  # kv_state + score_state
            dtype=state_dtype,
            compress_ratio=compress_ratio,
            prefix=f"{prefix}.state_cache",
        )

        # Save reference to static_forward_context for forward-time KV cache lookup.
        # get_current_vllm_config() is only available during __init__, not forward.
        self._static_forward_context = (
            vllm_config.compilation_config.static_forward_context
        )

        if self.head_dim == 512:
            assert not use_fp4_cache, (
                "MXFP4 cache is only supported for indexer (head=128)"
            )
            self._quant_block = 64
            self._token_stride = self.nope_head_dim + self.rope_head_dim * 2
            self._scale_dim = self.nope_head_dim // 64 + 1  # 7 real + 1 pad
        elif self.head_dim == 128:
            if use_fp4_cache:
                self._quant_block = MXFP4_BLOCK_SIZE
                self._token_stride = self.head_dim // 2
                self._scale_dim = self.head_dim // MXFP4_BLOCK_SIZE
            else:
                self._quant_block = 128
                self._token_stride = self.head_dim
                self._scale_dim = 4  # single float32 scale
        else:
            raise ValueError(
                f"Unsupported head_dim for fused quant+cache: {self.head_dim}"
            )
        # Note(Metax): This is used for maca indexer cache, set it to int8 as default
        self.use_int8_cache = (
            not use_fp4_cache
            and vllm_config.attention_config.indexer_kv_dtype == "auto"
        )
        # Note(Metax): This is used for compressed kv cache
        self.use_fp8_kvcache = vllm_config.cache_config.cache_dtype.startswith("fp8")
        if vllm_config.kernel_config.enable_jit_warmup:
            _SAVE_PARTIAL_STATES_KERNEL.register_warmup(
                head_dim=self.head_dim,
                compress_ratio=self.compress_ratio,
            )
            if self.head_dim == 128:
                _FUSED_KV_COMPRESS_NORM_ROPE_INSERT_INDEXER_TRITON_KERNEL.register_warmup(
                    vllm_config=vllm_config,
                    k_cache_prefix=self.k_cache_prefix,
                )

    def forward(
        self,
        # [num_tokens, 2 * self.coff * self.head_dim]
        kv_score: torch.Tensor,
        # [num_tokens]
        positions: torch.Tensor,
        rotary_emb,
    ) -> None:
        # Each of shape [num_tokens, coff * self.head_dim]
        # input bf16, output are fp32
        kv, score = kv_score.split(
            [self.coff * self.head_dim, self.coff * self.head_dim], dim=-1
        )

        # Get the metadata and handle dummy profiling run.
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if not isinstance(attn_metadata, dict):
            return

        state_metadata = cast(
            CompressorMetadata, attn_metadata[self.state_cache.prefix]
        )
        token_to_req_indices = state_metadata.token_to_req_indices
        slot_mapping = state_metadata.slot_mapping
        num_actual = slot_mapping.shape[0]
        block_table = state_metadata.block_table
        block_size = state_metadata.block_size

        # [num_blocks, block_size, kv_dim+score_dim], where kv_dim == score_dim
        state_cache = self.state_cache.kv_cache
        # kv_state stored in first half, score_state stored in second half
        state_width = state_cache.shape[-1] // 2
        # ---------------------------------------------
        # Note: Metax not support pdl
        pdl_kwargs: dict[str, Any] = {}

        _SAVE_PARTIAL_STATES_KERNEL(
            kv=kv,
            score=score,
            ape=self.ape,
            positions=positions,
            state_cache=state_cache,
            slot_mapping=slot_mapping,
            block_size=block_size,
            state_width=state_width,
            compress_ratio=self.compress_ratio,
            pdl_kwargs=pdl_kwargs,
        )

        # full graph cannot branch on per-step CPU metadata after capture
        if (
            current_platform.is_maca()
            and self.head_dim == 512
            and self.compress_ratio == 128
            and forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL
            and state_metadata.c128_boundary is False
        ):
            return

        self._compress_norm_rope_store(
            state_metadata,
            state_cache,
            state_width,
            positions,
            rotary_emb,
            attn_metadata,
            pdl_kwargs,
        )

    def _compress_norm_rope_store(
        self,
        state_metadata: CompressorMetadata,
        state_cache: torch.Tensor,
        state_width: int,
        positions: torch.Tensor,
        rotary_emb,
        attn_metadata: dict[str, Any],
        pdl_kwargs: dict[str, Any],
    ) -> None:
        """Compress saved states and write the MACA cache layout."""
        token_to_req_indices = state_metadata.token_to_req_indices
        slot_mapping = state_metadata.slot_mapping
        num_actual = slot_mapping.shape[0]
        block_table = state_metadata.block_table
        block_size = state_metadata.block_size

        # Fused: compress → RMSNorm → RoPE → cache-format quant → KV cache write.
        # RoPE requirements (kernel applies forward GPT-J style rotation):
        # - is_neox_style=False (interleaved pairs, NOT split-half)
        # - cos_sin_cache layout: [max_pos, rope_head_dim] with first half cos,
        #   second half sin (per-pair, length rope_head_dim // 2 each)
        # - applied to LAST rope_head_dim elements of head_dim
        # - position used: (positions // compress_ratio) * compress_ratio
        cos_sin_cache = rotary_emb.cos_sin_cache
        k_cache_metadata = cast(Any, attn_metadata[self.k_cache_prefix])
        k_cache_layer = self._static_forward_context[self.k_cache_prefix]
        kv_cache = k_cache_layer.kv_cache

        # MACA uses the local Triton launcher for both the full cache and the
        # indexer. The optional two-stage path remains ROCm-only.
        compress_norm_rope_store_fn: Any
        if self._use_two_stage_fused_compressor:
            # head=512 cr>=128 (no overlap): two-pass split compressor on the
            # prefill suffix, single-pass on the decode prefix.
            assert state_metadata.num_decode_tokens is not None
            compress_norm_rope_store_fn = compress_norm_rope_store_two_stage_triton
            extra_kwargs = {
                "num_decode_tokens": state_metadata.num_decode_tokens,
                "compress_scratch": self._compress_scratch,
            }
        else:
            # MetaX stores the main cache in BF16 or FP8 and the indexer in
            # INT8 by default, with FP8 selected explicitly.
            compress_norm_rope_store_fn = compress_norm_rope_store_triton
            extra_kwargs = {
                "use_int8_cache": self.use_int8_cache,
                "use_fp8_kvcache": self.use_fp8_kvcache,
            }

        compress_norm_rope_store_fn(
            state_cache=state_cache,
            num_actual=num_actual,
            token_to_req_indices=token_to_req_indices,
            positions=positions,
            slot_mapping=slot_mapping,
            block_table=block_table,
            block_size=block_size,
            state_width=state_width,
            cos_sin_cache=cos_sin_cache,
            kv_cache=kv_cache,
            k_cache_metadata=k_cache_metadata,
            pdl_kwargs=pdl_kwargs,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            compress_ratio=self.compress_ratio,
            overlap=self.overlap,
            use_fp4_cache=self.use_fp4_cache,
            rms_norm_weight=self.norm.weight,
            rms_norm_eps=self.rms_norm_eps,
            quant_block=self._quant_block,
            token_stride=self._token_stride,
            scale_dim=self._scale_dim,
            **extra_kwargs,
        )
