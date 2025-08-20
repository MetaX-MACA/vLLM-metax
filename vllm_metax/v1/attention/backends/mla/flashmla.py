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
from vllm_metax.attention.ops.flashmla import (flash_mla_with_kvcache,
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
        
    def _build_decode(self, block_table_tensor: torch.Tensor,
                      seq_lens: torch.Tensor) -> FlashMLADecodeMetadata:
        tile_scheduler_metadata, num_splits = \
            get_mla_metadata(
            seq_lens,
            self.num_q_heads,
            1, # MQA for the decode path
        )
        
        if self.runner.full_cuda_graph:
            # First time around (CUDAGraph capture), allocate the static buffer
            if self.cg_buf_tile_scheduler_metadata is None:
                self.cg_buf_tile_scheduler_metadata = tile_scheduler_metadata
                self.cg_buf_num_splits = num_splits
            else:
                assert self.cg_buf_num_splits is not None

                # Metadata per-SM, fixed size (#SMs, TileMetadataSize)
                assert (self.cg_buf_tile_scheduler_metadata.size() ==
                        tile_scheduler_metadata.size())
                self.cg_buf_tile_scheduler_metadata.\
                    copy_(tile_scheduler_metadata)
                tile_scheduler_metadata = self.cg_buf_tile_scheduler_metadata

                # Num splits is per-batch, varying size (batch_size,)
                n = num_splits.size(0)
                # make sure static buffer is large enough
                assert n <= self.cg_buf_num_splits.size(0)
                num_splits_view = self.cg_buf_num_splits[:n]
                num_splits_view.copy_(num_splits)
                self.cg_buf_num_splits[n:].fill_(0)  # fill the rest with 0s
                num_splits = num_splits_view

        return FlashMLADecodeMetadata(
            block_table=block_table_tensor,
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
            kv_sharing_target_layer_name: Optional[str],
            # MLA Specific Arguments
            **mla_args) -> None:
        super(FlashMLAImpl, self).__init__(num_heads, head_size, scale, num_kv_heads,
                         alibi_slopes, sliding_window, kv_cache_dtype,
                         blocksparse_params, logits_soft_cap, attn_type,
                         kv_sharing_target_layer_name, **mla_args)
        
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

        return self._v_up_proj(o)