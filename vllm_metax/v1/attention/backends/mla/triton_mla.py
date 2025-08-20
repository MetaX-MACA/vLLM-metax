# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Optional

import torch

from vllm.attention.backends.abstract import (AttentionType,
                                              is_quantized_kv_cache)
from vllm.attention.ops.triton_decode_attention import decode_attention_fwd
from vllm.logger import init_logger
from vllm.v1.attention.backends.mla.common import (MLACommonBackend,
                                                   MLACommonDecodeMetadata,
                                                   MLACommonImpl,
                                                   MLACommonMetadata,
                                                   MLACommonMetadataBuilder)

from vllm_metax.attention.backends.triton_mla import (load_config,
                                                            find_best_mla_para)

from vllm.model_executor.layers.linear import (LinearBase, 
                                               UnquantizedLinearMethod)

from flash_attn import flash_attn_varlen_func
from vllm import envs

logger = init_logger(__name__)

import os
# TODO: Configure environment variables temporarily. New versions do not need to be configured
os.environ['TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP'] = '1'
os.environ['TRITON_ENABLE_MACA_CHAIN_DOT_OPT'] = '1'

JSON_DATA = load_config()

class MetaxTritonMLABackend(MLACommonBackend):

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_VLLM_V1"

    @staticmethod
    def get_metadata_cls() -> type["MetaxTritonMLAMetadata"]:
        return MetaxTritonMLAMetadata

    @staticmethod
    def get_builder_cls() -> type["MetaxTritonMLAMetadataBuilder"]:
        return MetaxTritonMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["MetaxTritonMLAImpl"]:
        return MetaxTritonMLAImpl

@dataclass
class MetaxTritonMLADecodeMetadata(MLACommonDecodeMetadata):
    num_kv_splits: int
    num_stages: int

@dataclass
class MetaxTritonMLAMetadata(MLACommonMetadata[MetaxTritonMLADecodeMetadata]):
    pass

class MetaxTritonMLAMetadataBuilder(MLACommonMetadataBuilder[MetaxTritonMLAMetadata]):
    def _build_decode(self, block_table_tensor: torch.Tensor,
                      seq_lens: torch.Tensor) -> MetaxTritonMLADecodeMetadata:
        if seq_lens is not None:
            batch = seq_lens.shape[0]
            max_seq_len = int(seq_lens.max())
            num_kv_splits, num_stages = find_best_mla_para(JSON_DATA, batch, max_seq_len, 8)
        else:
            num_kv_splits = 4
            num_stages = 1
        return MetaxTritonMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens,
            num_kv_splits=num_kv_splits,
            num_stages=num_stages,
        )

class MetaxTritonMLAImpl(MLACommonImpl[MetaxTritonMLAMetadata]):

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
        super().__init__(num_heads, head_size, scale, num_kv_heads,
                         alibi_slopes, sliding_window, kv_cache_dtype,
                         blocksparse_params, logits_soft_cap, attn_type,
                         kv_sharing_target_layer_name, **mla_args)

        unsupported_features = [
            alibi_slopes, sliding_window, blocksparse_params, logits_soft_cap
        ]
        if any(unsupported_features):
            raise NotImplementedError(
                "TritonMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, blocksparse_params, "
                "logits_soft_cap")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "TritonMLAImpl")

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "TritonMLA V1 with FP8 KV cache not yet supported")

    def _forward_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
    ) -> torch.Tensor:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError("FP8 Triton MLA not yet supported")

        B = q_nope.shape[0]

        q = torch.cat([q_nope, q_pe], dim=-1)
        o = torch.zeros(B,
                        self.num_heads,
                        self.kv_lora_rank,
                        dtype=q.dtype,
                        device=q.device)

        # TODO(lucas) Allocate ahead of time
        attn_logits = torch.empty(
            (
                B,
                self.num_heads,
                # ┌------------------------  Metax Modification -------------------------┐
                attn_metadata.decode.num_kv_splits,
                # └------------------------- Metax Modification -------------------------┘
                
                # NOTE(lucas) idk why the +1 is here but sglang has it so we
                # just mirror that
                self.kv_lora_rank + 1,
            ),
            dtype=torch.float32,
            device=q.device,
        )

        # Add a head dim of 1
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.unsqueeze(2)
        kv_c_cache = kv_c_and_k_pe_cache[..., :self.kv_lora_rank]
        PAGE_SIZE = kv_c_and_k_pe_cache.size(1)

        # Run MQA
        decode_attention_fwd(q, kv_c_and_k_pe_cache, kv_c_cache, o,
                             attn_metadata.decode.block_table,
                             attn_metadata.decode.seq_lens, attn_logits,
                            # ┌------------------------  Metax Modification -------------------------┐
                             attn_metadata.decode.num_kv_splits,
                             attn_metadata.decode.num_stages,
                            # └------------------------- Metax Modification -------------------------┘
                             self.scale, PAGE_SIZE)

        return self._v_up_proj(o)

