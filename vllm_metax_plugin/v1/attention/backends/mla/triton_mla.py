# 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
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

from vllm_metax_plugin.attention.backends.triton_mla import (load_config,
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
    def _build_decode(self, input_positions: torch.Tensor,
                      block_table: torch.Tensor,
                      seq_lens: torch.Tensor) -> MetaxTritonMLADecodeMetadata:
        if seq_lens is not None:
            batch = seq_lens.shape[0]
            max_seq_len = int(seq_lens.max())
            num_kv_splits, num_stages = find_best_mla_para(JSON_DATA, batch, max_seq_len, 8)
        else:
            num_kv_splits = 4
            num_stages = 1

        return MetaxTritonMLADecodeMetadata(
            input_positions=input_positions,
            block_table=block_table,
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
            # MLA Specific Arguments
            **mla_args) -> None:
        super().__init__(num_heads, head_size, scale, num_kv_heads,
                         alibi_slopes, sliding_window, kv_cache_dtype,
                         blocksparse_params, logits_soft_cap, attn_type,
                         **mla_args)
        self.vllm_flash_attn_version = None
        self.flash_attn_varlen_func = flash_attn_varlen_func  # Metax Modification
        self._pad_v = True 

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

    # Modified on mla/common.py
    def _flash_attn_varlen_diff_headdims(self,
                                         q,
                                         k,
                                         v,
                                         return_softmax_lse=False,
                                         softmax_scale=None,
                                         **kwargs):
        maybe_padded_v = v
        if self._pad_v:
            maybe_padded_v = torch.nn.functional.pad(
                v, [0, q.shape[-1] - v.shape[-1]], value=0)

        attn_out = self.flash_attn_varlen_func(
            q=q,
            k=k,
            v=maybe_padded_v,
            return_attn_probs=return_softmax_lse, # Metax Modification
            softmax_scale=softmax_scale,
            **kwargs,
        )

        # Unpack the output if there is multiple results
        lse = None
        if isinstance(attn_out, tuple):
            attn_out, lse = attn_out[0], attn_out[1]

        # unpad if necessary
        if self._pad_v:
            attn_out = attn_out[..., :v.shape[-1]]

        # Remain consistent with old `flash_attn_varlen_func` where there
        # is only one output tensor if `return_softmax_lse` is False.
        if return_softmax_lse:
            return attn_out, lse
        return attn_out

    # Modified on mla/common.py
    def process_weights_after_loading(self, act_dtype: torch.dtype):

        def get_layer_weight(layer):
            WEIGHT_NAMES = ("weight", "qweight", "weight_packed")
            for attr in WEIGHT_NAMES:
                if hasattr(layer, attr):
                    return getattr(layer, attr)
            raise AttributeError(
                f"Layer '{layer}' has no recognized weight attribute:"
                f" {WEIGHT_NAMES}.")

        def get_and_maybe_dequant_weights(layer: LinearBase):
            if not isinstance(layer.quant_method, UnquantizedLinearMethod):
                # NOTE: This should only be used offline, since it's O(N^3)
                eye = torch.eye(layer.input_size_per_partition,
                                dtype=act_dtype,
                                device=get_layer_weight(layer).device)
                dequant_weights = layer.quant_method.apply(layer,
                                                           eye,
                                                           bias=None)
                del eye
                # standardize to (output, input)
                return dequant_weights.T
            # Metax Modification
            return layer.weight if not envs.MACA_VLLM_USE_TN_2_NN else layer.weight.T

        # we currently do not have quantized bmm's which are needed for
        # `W_UV` and `W_UK_T`, we we just store fp16/bf16 copies and perform
        # the bmm's in 16-bit, the extra memory overhead of this is fairly low
        kv_b_proj_weight = get_and_maybe_dequant_weights(self.kv_b_proj).T
        assert kv_b_proj_weight.shape == (
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim)), (
                f"{kv_b_proj_weight.shape=}, "
                f"{self.kv_lora_rank=}, "
                f"{self.num_heads=}, "
                f"{self.qk_nope_head_dim=}, "
                f"{self.v_head_dim=}")
        kv_b_proj_weight = kv_b_proj_weight.view(
            self.kv_lora_rank,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )

        W_UK, W_UV = kv_b_proj_weight.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # Convert from (L, N, V) to (N, L, V)
        self.W_UV = W_UV.transpose(0, 1)
        # Convert from (L, N, P) to (N, P, L)
        self.W_UK_T = W_UK.permute(1, 2, 0)

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
                attn_metadata.decode.num_kv_splits,
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
                             attn_metadata.decode.num_kv_splits,    # Metax Modification
                             attn_metadata.decode.num_stages,       # Metax Modification
                             self.scale, PAGE_SIZE)                 

        return self._v_up_proj_and_o_proj(o)
