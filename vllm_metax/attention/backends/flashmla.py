# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Type

import torch

from vllm.attention.backends.abstract import (AttentionType,
                                              is_quantized_kv_cache)
from vllm.attention.backends.flashmla import (FlashMLAImpl,
                                              FlashMLABackend,
                                              FlashMLAState,
                                              FlashMLAMetadata,
                                              FlashMLAMetadataBuilder,)
"""
Note:

Since we replaced flashmla ops with ours, all the related 
FlashMLABackend components are required to be replaced as well.
"""
from vllm_metax.attention.ops.flashmla import (flash_mla_with_kvcache,
                                                        get_mla_metadata,
                                                        is_flashmla_supported)



class MetaxFlashMLABackend(FlashMLABackend):

    @staticmethod
    def get_impl_cls() -> Type["FlashMLAImpl"]:
        return MetaxFlashMLAImpl

    @staticmethod
    def get_builder_cls() -> Type["FlashMLAMetadataBuilder"]:
        return MetaxFlashMLAMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["FlashMLAState"]:
        return MetaxFlashMLAState


class MetaxFlashMLAMetadataBuilder(FlashMLAMetadataBuilder):

    def build(self, seq_lens: List[int], query_lens: List[int],
              cuda_graph_pad_size: int, batch_size: int):
        m = super(FlashMLAMetadataBuilder, self).build(seq_lens, query_lens, cuda_graph_pad_size,
                          batch_size)

        if m.num_decode_tokens > 0:
            m.decode_tile_scheduler_metadata, m.decode_num_splits = \
                get_mla_metadata(   # Metax Modification
                m.seq_lens_tensor[m.num_prefills:],
                self.num_q_heads,
                1, # MQA for the decode path
            )

        return m


class MetaxFlashMLAState(FlashMLAState):

    @contextmanager
    def graph_capture(self, max_batch_size: int):
        # Run a dummy `get_mla_metadata` so we can get the right shapes
        self._graph_decoder_tile_scheduler_metadata, \
            self._graph_decode_num_splits = get_mla_metadata(   # Metax Modification
            torch.ones(
                max_batch_size, dtype=torch.int32, device=self.runner.device),
            self.num_q_heads,
            1, # MQA for the decode path
        )

        with super().graph_capture(max_batch_size):
            yield

        del self._graph_decoder_tile_scheduler_metadata
        del self._graph_decode_num_splits

    def graph_capture_get_metadata_for_batch(
            self, batch_size: int, is_encoder_decoder_model: bool = False):
        metadata = super(FlashMLAState, self).graph_capture_get_metadata_for_batch(
            batch_size, is_encoder_decoder_model)
        assert metadata.num_decode_tokens > 0

        decoder_tile_scheduler_metadata, decode_num_splits = get_mla_metadata(  # Metax Modifications
            self._graph_seq_lens[:batch_size],
            self.num_q_heads,
            1,  # MQA for the decode path
        )

        self._graph_decoder_tile_scheduler_metadata.copy_(
            decoder_tile_scheduler_metadata)
        self._graph_decode_num_splits[:batch_size + 1].copy_(decode_num_splits)

        metadata.decode_tile_scheduler_metadata=\
            self._graph_decoder_tile_scheduler_metadata
        metadata.decode_num_splits=\
            self._graph_decode_num_splits[:batch_size + 1]

        return metadata


class MetaxFlashMLAImpl(FlashMLAImpl):

    def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: int,
            alibi_slopes: Optional[List[float]],
            sliding_window: Optional[int],
            kv_cache_dtype: str,
            blocksparse_params: Optional[Dict[str, Any]],
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
                "FlashMLA with FP8 KV cache not yet supported")

    def _forward_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashMLAMetadata,
    ) -> torch.Tensor:
        assert kv_c_and_k_pe_cache.numel() > 0

        decode_meta = attn_metadata.decode_metadata
        assert decode_meta is not None

        q = torch.cat([q_nope, q_pe], dim=-1)\
            .unsqueeze(1) # Add seqlen dim of 1 (decode)

        o, _ = flash_mla_with_kvcache(  # Metax Modification
            q=q,
            k_cache=kv_c_and_k_pe_cache.unsqueeze(-2),  # Add head dim of 1
            block_table=decode_meta.block_tables,
            cache_seqlens=decode_meta.seq_lens_tensor,
            head_dim_v=self.kv_lora_rank,
            tile_scheduler_metadata=decode_meta.decode_tile_scheduler_metadata,
            num_splits=decode_meta.decode_num_splits,
            softmax_scale=self.scale,
            causal=True,
        )

        return self._v_up_proj_and_o_proj(o)
