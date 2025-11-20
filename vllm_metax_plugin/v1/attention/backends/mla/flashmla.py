# 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Optional

import torch

from vllm.attention.backends.abstract import (AttentionType,
                                              is_quantized_kv_cache)

"""
Note:

Since we replaced flashmla ops with ours, all the related 
FlashMLABackend components are required to be replaced as well.
"""
from vllm_metax_plugin.attention.ops.flashmla import (flash_mla_with_kvcache,
                                                        get_mla_metadata,
                                                        is_flashmla_supported)
from vllm.v1.attention.backends.mla.flashmla import (FlashMLAImpl,
                                                     FlashMLABackend,
                                                     FlashMLADecodeMetadata,
                                                     FlashMLAMetadata,
                                                     FlashMLAMetadataBuilder,)


class MetaxFlashMLABackend(FlashMLABackend):

    @staticmethod
    def get_builder_cls() -> type["FlashMLAMetadataBuilder"]:
        return MetaxFlashMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["FlashMLAImpl"]:
        return MetaxFlashMLAImpl


class MetaxFlashMLAMetadataBuilder(FlashMLAMetadataBuilder):
        
    def _build_decode(self, input_positions: torch.Tensor,
                      block_table: torch.Tensor,
                      seq_lens: torch.Tensor) -> FlashMLADecodeMetadata:
        tile_scheduler_metadata, num_splits = \
            get_mla_metadata(   # Metax Modification
            seq_lens,
            self.num_q_heads,
            1, # MQA for the decode path
        )

        return FlashMLADecodeMetadata(
            input_positions=input_positions,
            block_table=block_table,
            seq_lens=seq_lens,
            tile_scheduler_metadata=tile_scheduler_metadata,
            num_splits=num_splits,
        )


class MetaxFlashMLAImpl(FlashMLAImpl):

    def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: int,
            alibi_slopes: Optional[list[float]],
            sliding_window: Optional[int],
            kv_cache_dtype: str,
            blocksparse_params: Optional[dict[str, Any]],
            logits_soft_cap: Optional[float],
            attn_type: str,
            # MLA Specific Arguments
            **mla_args) -> None:
        super(FlashMLAImpl, self).__init__(num_heads, head_size, scale, num_kv_heads,
                         alibi_slopes, sliding_window, kv_cache_dtype,
                         blocksparse_params, logits_soft_cap, attn_type,
                         **mla_args)
        
        # Metax Modification
        assert is_flashmla_supported(), \
            "FlashMLA is not supported on this device"

        unsupported_features = [
            alibi_slopes, sliding_window, blocksparse_params, logits_soft_cap
        ]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, blocksparse_params, "
                "logits_soft_cap")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "FlashMLAImpl")

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "FlashMLA V1 with FP8 KV cache not yet supported")

    def _forward_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashMLAMetadata,
    ) -> torch.Tensor:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        q = torch.cat([q_nope, q_pe], dim=-1)\
            .unsqueeze(1) # Add seqlen dim of 1 (decode)

        # Metax Modification
        o, _ = flash_mla_with_kvcache(
            q=q,
            k_cache=kv_c_and_k_pe_cache.unsqueeze(-2),  # Add head dim of 1
            block_table=attn_metadata.decode.block_table,
            cache_seqlens=attn_metadata.decode.seq_lens,
            head_dim_v=self.kv_lora_rank,
            tile_scheduler_metadata=attn_metadata.decode.
            tile_scheduler_metadata,
            num_splits=attn_metadata.decode.num_splits,
            softmax_scale=self.scale,
            causal=True,
        )

        return self._v_up_proj_and_o_proj(o)
