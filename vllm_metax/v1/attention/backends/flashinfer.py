# SPDX-License-Identifier: Apache-2.0
"""Attention layer with FlashInfer."""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import TYPE_CHECKING, Any, Optional

import torch
from flashinfer import (BatchDecodeWithPagedKVCacheWrapper,
                        BatchPrefillWithPagedKVCacheWrapper,
                        MultiLevelCascadeAttentionWrapper)

import vllm.envs as envs
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionType)
from vllm.attention.layer import Attention
from vllm.config import (VllmConfig, get_current_vllm_config,
                         get_layers_from_vllm_config)
from vllm.logger import init_logger
from vllm.v1.attention.backends.flash_attn import use_cascade_attention
from vllm.v1.attention.backends.flashinfer import (FlashInferMetadata, 
                                                   FlashInferMetadataBuilder, 
                                                   get_per_layer_parameters,
                                                   infer_global_hyperparameters)
if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

FLASHINFER_WORKSPACE_BUFFER_SIZE = 256 * 1024 * 1024

logger = init_logger(__name__)


class MetaxFlashInferBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128, 256]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_VLLM_V1"

    @staticmethod
    def get_impl_cls() -> type[MetaxFlashInferImpl]:
        return MetaxFlashInferImpl

    @staticmethod
    def get_metadata_cls() -> type[MetaxFlashInferMetadata]:
        return MetaxFlashInferMetadata

    @staticmethod
    def get_builder_cls() -> type[MetaxFlashInferMetadataBuilder]:
        return MetaxFlashInferMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

@dataclass
class MetaxFlashInferMetadata(FlashInferMetadata):

    def __post_init__(self):
        # Refer to
        # https://github.com/flashinfer-ai/flashinfer/blob/3d55c71a62052c590c130897d3a3db49b14fcc34/include/flashinfer/utils.cuh#L157
        
        # ┌------------------------  Metax Modification -------------------------┐
        supported_head_sizes = MetaxFlashInferBackend.get_supported_head_sizes()
        # └------------------------- Metax Modification -------------------------┘

        if self.head_dim is not None and self.head_dim \
                not in supported_head_sizes:
            raise ValueError(
                f"Only {supported_head_sizes} are supported for head_dim,",
                f" received {self.head_dim}.")

class MetaxFlashInferMetadataBuilder(FlashInferMetadataBuilder):
    
    def _plan(self, attn_metadata: MetaxFlashInferMetadata):
        if self.global_hyperparameters is None:
            self.global_hyperparameters = infer_global_hyperparameters(
                get_per_layer_parameters(self.vllm_config))
        # ┌------------------------  Metax Modification -------------------------┐
        if attn_metadata.use_cascade and False: # not supported
        # └------------------------- Metax Modification -------------------------┘
            attn_metadata.cascade_wrapper = self._get_cascade_wrapper()
            attn_metadata.cascade_wrapper.plan(
                [attn_metadata.shared_qo_indptr, attn_metadata.qo_indptr],
                [
                    attn_metadata.shared_kv_page_indptr,
                    attn_metadata.paged_kv_indptr
                ],
                [
                    attn_metadata.shared_kv_page_indices,
                    attn_metadata.paged_kv_indices
                ],
                [
                    attn_metadata.shared_kv_last_page_len,
                    attn_metadata.paged_kv_last_page_len
                ],
                attn_metadata.num_qo_heads,
                attn_metadata.num_kv_heads,
                attn_metadata.head_dim,
                attn_metadata.page_size,
                causal=True,
                sm_scale=self.global_hyperparameters.sm_scale,
                window_left=self.global_hyperparameters.window_left,
                logits_soft_cap=self.global_hyperparameters.logits_soft_cap,
                q_data_type=attn_metadata.q_data_type,
            )
        else:
            # Regular attention (common case).
            # Decodes are at the front and prefills are at the back,
            # according to reorder_batch()
            if self._num_prefills > 0:
                # Decodes are first so prefills start after the last decode
                prefill_start = self._num_decodes
                attn_metadata.prefill_wrapper = self._get_prefill_wrapper()
                assert attn_metadata.qo_indptr[prefill_start:].shape[
                    0] == self._num_prefills + 1
                assert attn_metadata.paged_kv_indptr[prefill_start:].shape[
                    0] == self._num_prefills + 1
                assert attn_metadata.paged_kv_last_page_len[
                    prefill_start:].shape[0] == self._num_prefills
                # Since prefill_wrapper.run() will be called with
                # query[num_decode_tokens:] we need to adjust the qo_indptr
                # to be relative to the start of the prefill queries.
                qo_indptr = attn_metadata.qo_indptr[
                    prefill_start:] - attn_metadata.qo_indptr[prefill_start]
                attn_metadata.prefill_wrapper.plan(
                    qo_indptr,
                    attn_metadata.paged_kv_indptr[prefill_start:],
                    attn_metadata.paged_kv_indices,
                    attn_metadata.paged_kv_last_page_len[prefill_start:],
                    attn_metadata.num_qo_heads,
                    attn_metadata.num_kv_heads,
                    attn_metadata.head_dim,
                    attn_metadata.page_size,
                    causal=True,
                    sm_scale=self.global_hyperparameters.sm_scale,
                    window_left=self.global_hyperparameters.window_left,
                    logits_soft_cap=self.global_hyperparameters.
                    logits_soft_cap,
                    q_data_type=attn_metadata.q_data_type,
                    kv_data_type=attn_metadata.data_type,
                )

            if self._num_decodes > 0:
                attn_metadata.decode_wrapper = self._get_decode_wrapper()
                attn_metadata.decode_wrapper.plan(
                    attn_metadata.paged_kv_indptr[:self._num_decodes + 1],
                    attn_metadata.paged_kv_indices,
                    attn_metadata.paged_kv_last_page_len[:self._num_decodes],
                    attn_metadata.num_qo_heads,
                    attn_metadata.num_kv_heads,
                    attn_metadata.head_dim,
                    attn_metadata.page_size,
                    # Disable flashinfer's pos encoding and use vllm's rope.
                    pos_encoding_mode="NONE",
                    sm_scale=self.global_hyperparameters.sm_scale,
                    window_left=self.global_hyperparameters.window_left,
                    logits_soft_cap=self.global_hyperparameters.
                    logits_soft_cap,
                    q_data_type=attn_metadata.q_data_type,
                    kv_data_type=attn_metadata.data_type,
                )
                
    def use_cascade_attention(self, *args, **kwargs) -> bool:
        
        # ┌------------------------  Metax Modification -------------------------┐
        logger.warning_once(
                "Using cascade attention in MetaxFlashInfer is not supported yet")
        return False
        # └------------------------- Metax Modification -------------------------┘

        if self.kv_cache_spec.dtype != self.runner.model_config.dtype:
            # TODO: The cascade wrapper currently does not support setting
            # kv cache dtype to something different from query dtype.
            return False
        return use_cascade_attention(*args, **kwargs)

    def build(self, *args, **kwargs) -> MetaxFlashInferMetadata:
        origin =  FlashInferMetadataBuilder.build(*args, **kwargs)
        return MetaxFlashInferMetadata(**asdict(origin))

class MetaxFlashInferImpl(AttentionImpl):

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: MetaxFlashInferMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with FlashInfer.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache = [num_blocks, 2, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        if attn_metadata is None:
            # Profiling run.
            return output

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        if self.kv_sharing_target_layer_name is None:
            # Reshape the input keys and values and store them in the cache.
            # Skip this if sharing KV cache with an earlier attention layer.
            # NOTE(woosuk): Here, key and value are padded while slot_mapping is
            # not padded. However, we don't need to do key[:num_actual_tokens]
            # and value[:num_actual_tokens] because the reshape_and_cache_flash
            # op uses the slot_mapping's shape to determine the number of
            # actual tokens.
            torch.ops._C_cache_ops.reshape_and_cache_flash(
                key,
                value,
                kv_cache[:, 0],
                kv_cache[:, 1],
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )

        window_left = (self.sliding_window[0]
                       if self.sliding_window is not None else -1)

        # Inputs and outputs may be padded for CUDA graphs
        query = query[:num_actual_tokens]
        output_padded = output
        output = output[:num_actual_tokens]

        # ┌------------------------  Metax Modification -------------------------┐
        if attn_metadata.use_cascade and False:
        # └------------------------- Metax Modification -------------------------┘
            # Cascade attention (rare case).
            assert attn_metadata.cascade_wrapper is not None
            output.copy_(attn_metadata.cascade_wrapper.run(query, kv_cache))
            return output

        num_decode_tokens = attn_metadata.num_decode_tokens
        num_prefill_tokens = attn_metadata.num_prefill_tokens

        # Regular attention (common case).
        # Decodes are at the front and prefills are at the back,
        # according to reorder_batch()
        if prefill_wrapper := attn_metadata.prefill_wrapper:
            prefill_query = query[num_decode_tokens:]
            assert prefill_query.shape[0] == num_prefill_tokens
            assert prefill_wrapper is not None
            assert prefill_wrapper._causal
            assert prefill_wrapper._window_left == window_left
            assert prefill_wrapper._logits_soft_cap == (self.logits_soft_cap
                                                        or 0.0)
            assert prefill_wrapper._sm_scale == self.scale
            prefill_wrapper.run(
                prefill_query,
                kv_cache,
                k_scale=layer._k_scale_float,
                v_scale=layer._v_scale_float,
                out=output[num_decode_tokens:],
            )

        if decode_wrapper := attn_metadata.decode_wrapper:
            decode_query = query[:num_decode_tokens]
            assert decode_query.shape[0] == num_decode_tokens
            assert decode_wrapper is not None
            assert decode_wrapper._window_left == window_left
            assert decode_wrapper._logits_soft_cap == (self.logits_soft_cap
                                                       or 0.0)
            assert decode_wrapper._sm_scale == self.scale
            decode_wrapper.run(
                decode_query,
                kv_cache,
                k_scale=layer._k_scale_float,
                v_scale=layer._v_scale_float,
                out=output[:num_decode_tokens],
            )

        return output_padded