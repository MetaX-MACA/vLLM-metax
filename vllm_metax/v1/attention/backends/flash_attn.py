# SPDX-License-Identifier: Apache-2.0
"""Attention layer with FlashAttention."""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, ClassVar

import torch

from vllm import _custom_ops as ops
from vllm.attention.backends.abstract import (AttentionMetadata, AttentionType,
                                              is_quantized_kv_cache)
from vllm.attention.layer import Attention
from vllm.attention.ops.merge_attn_states import merge_attn_states

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.logger import init_logger

from vllm.v1.attention.backends.flash_attn import (FlashAttentionBackend,
                                                   FlashAttentionMetadata,
                                                   FlashAttentionImpl,
                                                   FlashAttentionMetadataBuilder,
                                                   CommonAttentionMetadata,
                                                   _get_sliding_window_configs,
                                                   use_cascade_attention,
                                                   make_local_attention_virtual_batches)

from vllm.v1.attention.backends.utils import AttentionMetadataBuilder
from vllm.v1.kv_cache_interface import AttentionSpec
from vllm.v1.worker.block_table import BlockTable

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.worker.gpu_input_batch import InputBatch
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner


from flash_attn import (flash_attn_varlen_func, flash_attn_with_kvcache)

logger = init_logger(__name__)

def flash_attn_supports_fp8() -> bool:
    return False

def get_flash_attn_version():
    return None

class MetaxFlashAttentionBackend(FlashAttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [32, 64, 80, 96, 112, 128, 160, 192, 224, 256]

    @staticmethod
    def get_impl_cls() -> type["MetaxFlashAttentionImpl"]:
        return MetaxFlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type["AttentionMetadata"]:
        return MetaxFlashAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type["MetaxFlashAttentionMetadataBuilder"]:
        return MetaxFlashAttentionMetadataBuilder

@dataclass(kw_only=True)
class MetaxFlashAttentionMetadata(FlashAttentionMetadata):
    # For handling prefill decode split
    num_decodes: int
    num_decode_tokens: int
    decode_query_start_loc: torch.Tensor
    decode_max_seq_len: int
    decode_seq_lens: torch.Tensor
    decode_block_table: torch.Tensor

    num_prefills: int
    num_prefill_tokens: int
    prefill_query_start_loc: torch.Tensor
    prefill_max_seq_len: int
    prefill_seq_lens: torch.Tensor
    prefill_block_table: torch.Tensor

class MetaxFlashAttentionMetadataBuilder(
        AttentionMetadataBuilder[MetaxFlashAttentionMetadata]):
    full_cudagraph_supported: ClassVar[bool] = True  # Decode-only

    def __init__(self, runner: "GPUModelRunner", kv_cache_spec: AttentionSpec,
                 block_table: BlockTable):
        model_config = runner.model_config
        compilation_config = runner.vllm_config.compilation_config

        self.runner = runner
        self.num_heads_q = model_config.get_num_attention_heads(
            runner.parallel_config)
        self.num_heads_kv = model_config.get_num_kv_heads(
            runner.parallel_config)
        self.headdim = model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size
        self.kv_cache_spec = kv_cache_spec
        self.block_table = block_table

        self.aot_schedule = (get_flash_attn_version() == 3)
        self.use_full_cuda_graph = compilation_config.full_cuda_graph
        self.scheduler_metadata = torch.zeros(self.runner.max_num_reqs + 1,
                                              dtype=torch.int32,
                                              device=self.runner.device)

        # Sliding window size to be used with the AOT scheduler will be
        # populated on first build() call.
        self.aot_sliding_window: Optional[tuple[int, int]] = None

    def reorder_batch(self, input_batch: "InputBatch",
                      scheduler_output: "SchedulerOutput") -> bool:
        # We now want to reorder the batch so that the "decode" requests are and
        # the front and the "prefill" requests are at the using the least amount
        # swaps possible. (NOTE for now we loosely use "decode" to mean requests
        # where attention is likely memory-bound and "prefill" to mean requests
        # where attention is likely compute-bound, TODO(lucas): figure out a
        # better naming here)
        decodes = []
        prefills = []
        num_decode_tokens = 0
        num_prefill_tokens = 0

        for i, req_id in enumerate(input_batch.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            # for now treat 1 scheduled token as "decode" even if its not,
            # we should update this to something like < 8 in the future but
            # currently the decode run only supports num_tokens = 1
            if num_tokens == 1:
                decodes.append(i)
                num_decode_tokens += num_tokens
            else:
                prefills.append(i)
                num_prefill_tokens += num_tokens

        # We hope that this is fairly minimal since decodes
        # should be around for a number of iterations so hopefully they are
        # relatively stationary (and new request are generally appended to the
        # persistent batch so already should be at the back)
        # To achieve this we loop over the decodes in descending order and
        # the prefills in ascending order. We swap decodes from the  "back"
        # i.e. past where the last decode should be in the reodorered with
        # prefills from the front of the batch.
        # `decodes` and `prefills` are already in ascending order just based on
        # the above loop
        num_decodes = len(decodes)
        num_prefills = len(prefills)
        modified_batch = False

        for i in range(1, min(num_decodes, num_prefills) + 1):
            # If the decode is at the "back" of the batch, i, we can swap it
            # with the prefill closest to the front of the batch
            decode_idx = decodes[num_decodes - i]
            if decode_idx < num_decodes:
                break

            input_batch.swap_states(prefills[i - 1], decode_idx)
            modified_batch = True

        # Save for next `build` call
        # TODO(lucas): this is a bit of a hack, we should probably have a
        # better way of doing this
        self._num_decodes = num_decodes
        self._num_prefills = num_prefills
        self._num_decode_tokens = num_decode_tokens
        self._num_prefill_tokens = num_prefill_tokens

        return modified_batch

    def build_for_cudagraph_capture(
            self, common_attn_metadata: CommonAttentionMetadata) -> MetaxFlashAttentionMetadata:
        """
        This method builds the metadata for full cudagraph capture.
        Currently, only decode is supported for full cudagraphs with MLA.
        """
        m = common_attn_metadata
        assert m.num_reqs == m.num_actual_tokens, \
            "MLA only supports decode-only full CUDAGraph capture. " \
            "Make sure all cudagraph capture sizes <= max_num_seq."

        m.max_query_len = 1  # decode-only

        # Update state usually set in reorder_batch.
        self._num_decodes = m.num_reqs
        self._num_decode_tokens = m.num_actual_tokens
        self._num_prefills = 0
        self._num_prefill_tokens = 0
        return self.build(0, m)

    def build(
        self, common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata
    ) -> MetaxFlashAttentionMetadata:
        # ┌------------------------  Metax Modification -------------------------┐
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        assert self._num_decodes + self._num_prefills == num_reqs
        assert (self._num_decode_tokens +
                self._num_prefill_tokens == num_actual_tokens)
        # └------------------------- Metax Modification -------------------------┘

        max_seq_len = int(self.runner.seq_lens_np[:num_reqs].max())
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table = self.block_table
        block_table_tensor = block_table.get_device_tensor()[:num_reqs]

        block_table.slot_mapping[:num_actual_tokens].copy_(
            block_table.slot_mapping_cpu[:num_actual_tokens],
            non_blocking=True)
        # Fill unused with -1. Needed for reshape_and_cache in full cuda graph
        # mode.
        block_table.slot_mapping[num_actual_tokens:].fill_(-1)

        slot_mapping = block_table.slot_mapping[:num_actual_tokens]

        # ┌------------------------  Metax Modification -------------------------┐
        # For handling prefill decode split
        if self._num_decodes > 0:
            decode_max_seq_len = int(self.runner.seq_lens_np[:self._num_decodes].max())
            decode_query_start_loc = common_attn_metadata.query_start_loc[:self._num_decodes + 1]
            decode_seq_lens = common_attn_metadata.seq_lens[:self._num_decodes]
            decode_block_table_tensor = block_table.get_device_tensor()[:self._num_decodes]
        else:
            decode_max_seq_len = 0
            decode_query_start_loc = None
            decode_seq_lens = None
            decode_block_table_tensor = None

        if self._num_prefills > 0:
            prefill_max_seq_len = int(self.runner.seq_lens_np[self._num_decodes:num_reqs].max())
            prefill_query_start_loc = (common_attn_metadata.query_start_loc[self._num_decodes:num_reqs + 1] -
                                       common_attn_metadata.query_start_loc[self._num_decodes])
            prefill_seq_lens = common_attn_metadata.seq_lens[self._num_decodes:num_reqs]
            prefill_block_table_tensor = block_table.get_device_tensor()[self._num_decodes:num_reqs]
        else:
            prefill_max_seq_len = 0
            prefill_query_start_loc = None
            prefill_seq_lens = None
            prefill_block_table_tensor = None
        # └------------------------- Metax Modification -------------------------┘
        
        if self.aot_sliding_window is None:
            self.aot_sliding_window = (-1, -1)
            # For the AOT scheduler we need the sliding window value to be
            # constant for all layers to. We have to populate this on the first
            # build() call so the layers are constructed (cannot populate)
            # in __init__.
            if self.aot_schedule:
                sliding_window_configs = _get_sliding_window_configs(
                    self.runner.vllm_config)
                if len(sliding_window_configs) == 1:
                    sliding_window_config = sliding_window_configs.pop()
                    if sliding_window_config is not None:
                        self.aot_sliding_window = sliding_window_config
                elif len(sliding_window_configs) > 1:
                    self.aot_schedule = False

        def schedule(batch_size, cu_query_lens, max_query_len, seqlens,
                     max_seq_len, causal):
            # ┌------------------------  Metax Modification -------------------------┐
            # if self.aot_schedule:
            #     return get_scheduler_metadata(
            #         batch_size=batch_size,
            #         max_seqlen_q=max_query_len,
            #         max_seqlen_k=max_seq_len,
            #         cache_seqlens=seqlens,
            #         num_heads_q=self.num_heads_q,
            #         num_heads_kv=self.num_heads_kv,
            #         headdim=self.headdim,
            #         page_size=self.page_size,
            #         cu_seqlens_q=cu_query_lens,
            #         causal=causal,
            #         window_size=self.aot_sliding_window,
            #     )
            # └------------------------- Metax Modification -------------------------┘
            return None

        # for local attention
        local_attn_metadata = None
        if self.runner.attention_chunk_size is not None:
            seqlens_q_local_np, virt_q_cu_seqlens_np, virt_k_seqlens_np, \
                virt_block_table_tensor = make_local_attention_virtual_batches(
                    self.runner.attention_chunk_size,
                    self.runner.query_start_loc_np[:num_reqs + 1],
                    self.runner.seq_lens_np[:num_reqs],
                    block_table_tensor,
                    self.block_size,
                )
            local_query_start_loc = torch.from_numpy(virt_q_cu_seqlens_np).to(
                self.runner.device, non_blocking=True)
            local_seqused_k = torch.from_numpy(virt_k_seqlens_np).to(
                self.runner.device, non_blocking=True)
            local_max_query_len = seqlens_q_local_np.max()
            local_max_seq_len = virt_k_seqlens_np.max()
            local_scheduler_metadata = schedule(
                batch_size=local_query_start_loc.shape[0] - 1,
                cu_query_lens=local_query_start_loc,
                max_query_len=local_max_query_len,
                seqlens=local_seqused_k,
                max_seq_len=local_max_seq_len,
                causal=True)

            local_attn_metadata = FlashAttentionMetadata.LocalAttentionMetadata(
                local_query_start_loc=local_query_start_loc,
                local_seqused_k=local_seqused_k,
                local_block_table=virt_block_table_tensor,
                local_max_query_len=local_max_query_len,
                local_max_seq_len=local_max_seq_len,
                local_scheduler_metadata=local_scheduler_metadata,
            )

        use_cascade = common_prefix_len > 0

        if use_cascade:
            cu_prefix_query_lens = torch.tensor([0, num_actual_tokens],
                                                dtype=torch.int32,
                                                device=self.runner.device)
            prefix_kv_lens = torch.tensor([common_prefix_len],
                                          dtype=torch.int32,
                                          device=self.runner.device)
            suffix_kv_lens = (self.runner.seq_lens_np[:num_reqs] -
                              common_prefix_len)
            suffix_kv_lens = torch.from_numpy(suffix_kv_lens).to(
                self.runner.device)
            prefix_scheduler_metadata = schedule(
                batch_size=1,
                cu_query_lens=cu_prefix_query_lens,
                max_query_len=num_actual_tokens,
                seqlens=prefix_kv_lens,
                max_seq_len=common_prefix_len,
                causal=False)
            scheduler_metadata = schedule(batch_size=num_reqs,
                                          cu_query_lens=query_start_loc,
                                          max_query_len=max_query_len,
                                          seqlens=suffix_kv_lens,
                                          max_seq_len=max_seq_len -
                                          common_prefix_len,
                                          causal=True)
        else:
            cu_prefix_query_lens = None
            prefix_kv_lens = None
            suffix_kv_lens = None
            prefix_scheduler_metadata = None
            scheduler_metadata = schedule(batch_size=num_reqs,
                                          cu_query_lens=query_start_loc,
                                          max_query_len=max_query_len,
                                          seqlens=seq_lens,
                                          max_seq_len=max_seq_len,
                                          causal=True)

        attn_metadata = MetaxFlashAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            # ┌------------------------  Metax Modification -------------------------┐
            # For handling prefill decode split
            num_decodes=self._num_decodes,
            num_decode_tokens=self._num_decode_tokens,
            decode_query_start_loc=decode_query_start_loc,
            decode_max_seq_len=decode_max_seq_len,
            decode_seq_lens=decode_seq_lens,
            decode_block_table=decode_block_table_tensor,
            num_prefills=self._num_prefills,
            num_prefill_tokens=self._num_prefill_tokens,
            prefill_query_start_loc=prefill_query_start_loc,
            prefill_max_seq_len=prefill_max_seq_len,
            prefill_seq_lens=prefill_seq_lens,
            prefill_block_table=prefill_block_table_tensor,
            # └------------------------- Metax Modification -------------------------┘
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            scheduler_metadata=scheduler_metadata,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            local_attn_metadata=local_attn_metadata,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
        )
        return attn_metadata

    def can_run_in_cudagraph(
            self, common_attn_metadata: CommonAttentionMetadata) -> bool:
        return common_attn_metadata.max_query_len == 1

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return use_cascade_attention(*args, **kwargs)

class MetaxFlashAttentionImpl(FlashAttentionImpl):

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[list[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: Optional[str] = None,
        use_irope: bool = False,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError(
                "FlashAttention does not support block-sparse attention.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        
        # ┌------------------------  Metax Modification -------------------------┐
        support_head_sizes = MetaxFlashAttentionBackend.get_supported_head_sizes()
        # └------------------------- Metax Modification -------------------------┘

        if head_size not in support_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by FlashAttention. "
                f"Supported head sizes are: {support_head_sizes}. "
                "Set VLLM_USE_V1=0 to use another attention backend.")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "FlashAttentionImpl")
        self.use_irope = use_irope
        self.vllm_flash_attn_version = get_flash_attn_version()
        if is_quantized_kv_cache(self.kv_cache_dtype) \
            and not flash_attn_supports_fp8():
            raise NotImplementedError(
                "FlashAttention does not support fp8 kv-cache on this device.")

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: MetaxFlashAttentionMetadata,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        NOTE: FP8 quantization, flash-attn expect the size of
              {q,k,v}_descale to be (num_sequences, num_kv_heads).
              We use torch's .expand() to avoid duplicating values
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
        key_cache, value_cache = kv_cache.unbind(0)

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
                key_cache,
                value_cache,
                attn_metadata.slot_mapping,
                self.kv_cache_dtype,
                layer._k_scale,
                layer._v_scale,
            )

        if self.kv_cache_dtype.startswith("fp8"):
            key_cache = key_cache.view(torch.float8_e4m3fn)
            value_cache = value_cache.view(torch.float8_e4m3fn)
            num_tokens, num_heads, head_size = query.shape
            query, _ = ops.scaled_fp8_quant(
                query.reshape(
                    (num_tokens, num_heads * head_size)).contiguous(),
                layer._q_scale)
            query = query.reshape((num_tokens, num_heads, head_size))

        # Compute attention and update output up to `num_actual_tokens`.
        use_local_attn = \
            (self.use_irope and attn_metadata.local_attn_metadata is not None)
        
        # ┌------------------------  Metax Modification -------------------------┐
        # For handling prefill decode split
        if not attn_metadata.use_cascade and not use_local_attn:
            num_decode_tokens = attn_metadata.num_decode_tokens
            if attn_metadata.num_prefills > 0:
                cu_prefix_kv_lens = torch.tensor([0] + attn_metadata.prefill_seq_lens.tolist(), device=attn_metadata.prefill_seq_lens.device, dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)
                output[num_decode_tokens:num_actual_tokens] = flash_attn_varlen_func(
                    q=query[num_decode_tokens:num_actual_tokens],
                    k=key_cache,
                    v=value_cache,
                    block_table=attn_metadata.prefill_block_table,
                    cu_seqlens_q=attn_metadata.prefill_query_start_loc,
                    cu_seqlens_k=cu_prefix_kv_lens,
                    max_seqlen_q=attn_metadata.max_query_len,
                    max_seqlen_k=attn_metadata.prefill_max_seq_len,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=self.sliding_window,
                    alibi_slopes=self.alibi_slopes,
                    softcap=self.logits_soft_cap,
                )
            if attn_metadata.num_decodes > 0:
                # Use flash_attn_with_kvcache for normal decoding.
                decode_query = query[:num_decode_tokens]
                output[:num_decode_tokens] = flash_attn_with_kvcache(
                    q=decode_query.unsqueeze(1),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    block_table=attn_metadata.decode_block_table,
                    cache_seqlens=attn_metadata.decode_seq_lens,
                    softmax_scale=self.scale,
                    causal=True,
                    window_size=self.sliding_window,
                    alibi_slopes=self.alibi_slopes,
                    softcap=self.logits_soft_cap,
                ).squeeze(1)
            return output
        # └------------------------- Metax Modification -------------------------┘

        if not attn_metadata.use_cascade or use_local_attn:
            if use_local_attn:
                assert attn_metadata.local_attn_metadata is not None
                local_metadata = attn_metadata.local_attn_metadata
                cu_seqlens_q = local_metadata.local_query_start_loc
                seqused_k = local_metadata.local_seqused_k
                max_seqlen_q = local_metadata.local_max_query_len
                max_seqlen_k = local_metadata.local_max_seq_len
                block_table = local_metadata.local_block_table
                scheduler_metadata = local_metadata.local_scheduler_metadata
            else:
                cu_seqlens_q = attn_metadata.query_start_loc
                seqused_k = attn_metadata.seq_lens
                max_seqlen_q = attn_metadata.max_query_len
                max_seqlen_k = attn_metadata.max_seq_len
                block_table = attn_metadata.block_table
                scheduler_metadata = attn_metadata.scheduler_metadata
                
            # ┌------------------------  Metax Modification -------------------------┐
            # descale_shape = (cu_seqlens_q.shape[0] - 1, key.shape[1])
            cu_prefix_kv_lens = torch.tensor([0] + seqused_k.tolist(), seqused_k.device, 
                                             dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)
            # └------------------------- Metax Modification -------------------------┘

            output[:num_actual_tokens] = flash_attn_varlen_func(
                q=query[:num_actual_tokens],
                k=key_cache,
                v=value_cache,
                # ┌------------------------  Metax Modification -------------------------┐
                # out=output[:num_actual_tokens],
                # └------------------------- Metax Modification -------------------------┘
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                # ┌------------------------  Metax Modification -------------------------┐
                # seqused_k=seqused_k,
                # └------------------------- Metax Modification -------------------------┘
                cu_seqlens_k=cu_prefix_kv_lens,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                window_size=self.sliding_window,
                block_table=block_table,
                softcap=self.logits_soft_cap,
                # ┌------------------------  Metax Modification -------------------------┐
                # scheduler_metadata=scheduler_metadata,
                # fa_version=self.vllm_flash_attn_version,
                # q_descale=layer._q_scale.expand(descale_shape),
                # k_descale=layer._k_scale.expand(descale_shape),
                # v_descale=layer._v_scale.expand(descale_shape),
                # └------------------------- Metax Modification -------------------------┘
            )
            return output

        assert not use_local_attn, (
            "Cascade attention does not support local attention.")
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
            fa_version=self.vllm_flash_attn_version,
            prefix_scheduler_metadata=attn_metadata.prefix_scheduler_metadata,
            suffix_scheduler_metadata=attn_metadata.scheduler_metadata,
            q_descale=layer._q_scale,
            k_descale=layer._k_scale,
            v_descale=layer._v_scale,
        )
        return output

def cascade_attention(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    cu_query_lens: torch.Tensor,
    max_query_len: int,
    cu_prefix_query_lens: torch.Tensor,
    prefix_kv_lens: torch.Tensor,
    suffix_kv_lens: torch.Tensor,
    max_kv_len: int,
    softmax_scale: float,
    alibi_slopes: Optional[torch.Tensor],
    sliding_window: tuple[int, int],
    logits_soft_cap: float,
    block_table: torch.Tensor,
    common_prefix_len: int,
    fa_version: int,
    prefix_scheduler_metadata: Optional[torch.Tensor] = None,
    suffix_scheduler_metadata: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    assert alibi_slopes is None, ("Cascade attention does not support ALiBi.")
    # TODO: Support sliding window.
    assert sliding_window == (-1, -1), (
        "Cascade attention does not support sliding window.")

    num_tokens = query.shape[0]
    block_size = key_cache.shape[-3]
    assert common_prefix_len % block_size == 0
    num_common_kv_blocks = common_prefix_len // block_size
    assert num_common_kv_blocks > 0
    descale_shape = (cu_prefix_query_lens.shape[0] - 1, key_cache.shape[-2])
    
    # Process shared prefix.

    # ┌------------------------  Metax Modification -------------------------┐
    cu_prefix_kv_lens = torch.tensor([0] + prefix_kv_lens.tolist(), 
                                     device=prefix_kv_lens.device, 
                                     dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)
    # └------------------------- Metax Modification -------------------------┘

    prefix_output, prefix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_prefix_query_lens,
        # ┌------------------------  Metax Modification -------------------------┐
        cu_seqlens_k=cu_prefix_kv_lens,
        # └------------------------- Metax Modification -------------------------┘
        max_seqlen_q=num_tokens,
        max_seqlen_k=common_prefix_len,
        softmax_scale=softmax_scale,
        causal=False,
        window_size=sliding_window,
        block_table=block_table[:1],
        softcap=logits_soft_cap,
        return_softmax_lse=True,
        # ┌------------------------  Metax Modification -------------------------┐
        # scheduler_metadata=prefix_scheduler_metadata,
        # fa_version=fa_version,
        # q_descale=q_descale.expand(descale_shape)
        # if q_descale is not None else None,
        # k_descale=k_descale.expand(descale_shape)
        # if k_descale is not None else None,
        # v_descale=v_descale.expand(descale_shape)
        # if v_descale is not None else None,
        # └------------------------- Metax Modification -------------------------┘
    )

    descale_shape = (cu_query_lens.shape[0] - 1, key_cache.shape[-2])
    
    # Process suffix per query.
    # ┌------------------------  Metax Modification -------------------------┐
    cu_suffix_kv_lens = torch.tensor([0] + suffix_kv_lens.tolist(), 
                                     device=suffix_kv_lens.device, 
                                     dtype=torch.int32).cumsum(dim=0, dtype=torch.int32)
    # └------------------------- Metax Modification -------------------------┘

    suffix_output, suffix_lse = flash_attn_varlen_func(
        q=query,
        k=key_cache,
        v=value_cache,
        cu_seqlens_q=cu_query_lens,
        # ┌------------------------  Metax Modification -------------------------┐
        cu_seqlens_k=cu_suffix_kv_lens,
        # └------------------------- Metax Modification -------------------------┘
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len - common_prefix_len,
        softmax_scale=softmax_scale,
        causal=True,
        window_size=sliding_window,
        block_table=block_table[:, num_common_kv_blocks:],
        softcap=logits_soft_cap,
        return_softmax_lse=True,
        # ┌------------------------  Metax Modification -------------------------┐
        # scheduler_metadata=suffix_scheduler_metadata,
        # fa_version=fa_version,
        # q_descale=q_descale.expand(descale_shape)
        # if q_descale is not None else None,
        # k_descale=k_descale.expand(descale_shape)
        # if k_descale is not None else None,
        # v_descale=v_descale.expand(descale_shape)
        # if v_descale is not None else None,
        # └------------------------- Metax Modification -------------------------┘
    )

    # Merge prefix and suffix outputs, and store the result in output.
    merge_attn_states(output, prefix_output, prefix_lse, suffix_output,
                      suffix_lse)
