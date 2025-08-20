# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import torch

from vllm.attention.backends.abstract import (AttentionType,
                                              is_quantized_kv_cache)
from vllm.attention.backends.mla.common import (MLACommonBackend,
                                                MLACommonImpl,
                                                MLACommonMetadata)
from vllm.attention.ops.triton_decode_attention import decode_attention_fwd

import json
import os

# TODO: Configure environment variables temporarily. New versions do not need to be configured
os.environ['TRITON_ENABLE_MACA_OPT_MOVE_DOT_OPERANDS_OUT_LOOP'] = '1'
os.environ['TRITON_ENABLE_MACA_CHAIN_DOT_OPT'] = '1'

def load_config():
    # Load JSON data from the file
    json_path = config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", "tp8_merge.json")
    with open(json_path, 'r') as file:
        data = json.load(file)
    return data

JSON_DATA = load_config()

def find_best_mla_para(json_data, batch_size, input_len, tp_size):
    best_match = None
    best_batch_size_diff = float('inf')
    best_input_len_diff = float('inf')
    
    for entry in json_data:
        if entry["BS"] == batch_size and entry["L"] == input_len:
            return entry["num_kv_splits"], entry['num_stages']
        batch_size_diff = abs(entry["BS"] - batch_size)
        input_len_diff = abs(entry["L"] - input_len)
        
        # Check if this is a better match than the current best match
        if batch_size_diff < best_batch_size_diff or (batch_size_diff == best_batch_size_diff and input_len_diff < best_input_len_diff):
            best_match = entry
            best_batch_size_diff = batch_size_diff
            best_input_len_diff = input_len_diff
    
    # If a match was found, return the best_kv_splits, otherwise return None
    return best_match["num_kv_splits"],best_match["num_stages"]


class MetaxTritonMLABackend(MLACommonBackend):

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA"

    @staticmethod
    def get_impl_cls() -> Type["MetaxTritonMLAImpl"]:
        return MetaxTritonMLAImpl
    
    @staticmethod
    def get_metadata_cls() -> Type["MetaxTritonMLAMetadata"]:
        return MetaxTritonMLAMetadata

@dataclass
class MetaxTritonMLAMetadata(MLACommonMetadata):
    num_kv_splits: int = 4  # TODO: heuristic
    num_stages: int = 1

    @property
    def decode_metadata(self):
        if self.num_decode_tokens == 0:
            return None
        if self._cached_decode_metadata is not None:
            return self._cached_decode_metadata
        
        decode_metadata = super().decode_metadata
        
        if decode_metadata is not None:
            if decode_metadata.seq_lens_tensor is not None:
                batch = decode_metadata.seq_lens_tensor.shape[0]
                max_seq_len = int(decode_metadata.seq_lens_tensor.max())
                num_kv_splits, num_stages = find_best_mla_para(JSON_DATA, batch, max_seq_len, 8)
            else:
                num_kv_splits = self.num_kv_splits
                num_stages = self.num_stages
            decode_metadata.num_kv_splits = num_kv_splits
            decode_metadata.num_stages = num_stages
        return decode_metadata

class MetaxTritonMLAImpl(MLACommonImpl[MetaxTritonMLAMetadata]):

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
        super().__init__(num_heads, head_size, scale, num_kv_heads,
                         alibi_slopes, sliding_window, kv_cache_dtype,
                         blocksparse_params, logits_soft_cap, attn_type,
                         **mla_args)

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
                "TritonMLA with FP8 KV cache not yet supported")

    def _forward_decode(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MetaxTritonMLAMetadata,
    ) -> torch.Tensor:
        assert kv_c_and_k_pe_cache.numel() > 0

        decode_meta = attn_metadata.decode_metadata
        assert decode_meta is not None
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
                decode_meta.num_kv_splits, # Metax Modification
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
                             decode_meta.block_tables,
                             decode_meta.seq_lens_tensor, attn_logits,
                             decode_meta.num_kv_splits, decode_meta.num_stages, self.scale, PAGE_SIZE) # Metax Modification

        return self._v_up_proj(o)
