# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass
from typing import ClassVar, Optional, Union

from flashinfer import BatchDecodeWithPagedKVCacheWrapper
import torch
from flashinfer.mla import BatchMLAPagedAttentionWrapper

from vllm.attention.backends.abstract import (AttentionLayer, AttentionType,
                                              is_quantized_kv_cache)
from vllm.logger import init_logger
from vllm_metax.v1.attention.backends.mla.common import (
    MLACommonBackend, MLACommonDecodeMetadata, MLACommonImpl,
    MLACommonMetadata, MLACommonMetadataBuilder)
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.utils import cdiv, is_pin_memory_available
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport, AttentionMetadataBuilder, CommonAttentionMetadata,
    get_kv_cache_layout, get_per_layer_parameters,
    infer_global_hyperparameters, split_decodes_and_prefills)

# yapf: enable
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

FLASHINFER_MLA_WORKSPACE_BUFFER_SIZE = 128 * 1024 * 1024


class MacaFlashInferMLABackend(MLACommonBackend):

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_MLA"

    @staticmethod
    def get_metadata_cls() -> type["FlashInferMLAMetadata"]:
        return FlashInferMLAMetadata

    @staticmethod
    def get_builder_cls() -> type["FlashInferMLAMetadataBuilder"]:
        return FlashInferMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["FlashInferMLAImpl"]:
        return FlashInferMLAImpl


@dataclass
class FlashInferMLADecodeMetadata(MLACommonDecodeMetadata):
    decode_wrapper: Optional[BatchMLAPagedAttentionWrapper] = None
    qo_indptr_gpu: Optional[torch.Tensor] = None
    paged_kv_indptr_gpu: Optional[torch.Tensor] = None


@dataclass
class FlashInferMLAMetadata(MLACommonMetadata[FlashInferMLADecodeMetadata]):
    pass


class FlashInferMLAMetadataBuilder(
        MLACommonMetadataBuilder[FlashInferMLAMetadata]):
    cudagraph_support: ClassVar[AttentionCGSupport] = \
        AttentionCGSupport.UNIFORM_BATCH

    reorder_batch_threshold: int = 1

    def __init__(self, kv_cache_spec: AttentionSpec, layer_names: list[str],
                 vllm_config: VllmConfig, device: torch.device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.cache_config = vllm_config.cache_config
        self.model_config = vllm_config.model_config
        self.compilation_config = vllm_config.compilation_config

        self._workspace_buffer = None
        self._decode_wrapper = None  # Wrapper for decode (general shape)

        max_num_pages_per_req = cdiv(self.model_config.max_model_len,
                                     self.kv_cache_spec.block_size)
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        max_num_pages = max_num_reqs * max_num_pages_per_req

        self.enable_cuda_graph = (self.compilation_config.cudagraph_mode.\
            decode_mode() == CUDAGraphMode.FULL)
        if self.enable_cuda_graph:
            # For full cudagraph capture, one `decode_wrapper` for each batch
            # size is needed for FlashInfer.
            self._decode_wrappers_cudagraph: dict[
                int, BatchMLAPagedAttentionWrapper] = {}
            self._decode_cudagraph_max_bs = min(
                max_num_reqs, self.compilation_config.max_capture_size)

        self.num_qo_heads = self.model_config.get_num_attention_heads(
            vllm_config.parallel_config)
        self.num_kv_heads = self.kv_cache_spec.num_kv_heads
        self.head_dim = self.kv_cache_spec.head_size
        MacaFlashInferMLABackend.validate_head_size(self.head_dim)
        self.page_size = self.kv_cache_spec.block_size

        self.cache_dtype = self.cache_config.cache_dtype
        # Maca do not support fp8 kv cache
        assert self.kv_cache_spec.dtype == self.model_config.dtype
        self.kv_cache_dtype = self.kv_cache_spec.dtype

        self.q_data_type = self.model_config.dtype

        # Preparing persistent buffers (device-side)
        self.qo_indptr = torch.arange(0,
                                      max_num_reqs + 1,
                                      dtype=torch.int32,
                                      device=self.device)
        self.paged_kv_indptr = torch.zeros(max_num_reqs + 1,
                                           dtype=torch.int32,
                                           device=self.device)
        self.paged_kv_indices = torch.zeros(
            max_num_pages,  # max num pages possible
            dtype=torch.int32,
            device=self.device)
        self.paged_kv_len_arr = torch.zeros(max_num_reqs,
                                            dtype=torch.int32,
                                            device=self.device)

        # host-side buffer
        pin_memory = is_pin_memory_available()
        self.qo_indptr_cpu = torch.arange(0,
                                          max_num_reqs + 1,
                                          dtype=torch.int32,
                                          device="cpu",
                                          pin_memory=pin_memory)
        self.paged_kv_indptr_cpu = torch.zeros(max_num_reqs + 1,
                                               dtype=torch.int32,
                                               device="cpu",
                                               pin_memory=pin_memory)
        self.paged_kv_indptr_np = self.paged_kv_indptr_cpu.numpy()
        self.paged_kv_indptr_buffer = torch.zeros_like(
            self.paged_kv_indptr_cpu, pin_memory=pin_memory)
        self.paged_kv_indices_cpu = torch.zeros(max_num_pages,
                                                dtype=torch.int32,
                                                device="cpu",
                                                pin_memory=pin_memory)
        self.paged_kv_len_arr_cpu = torch.zeros(max_num_reqs,
                                                dtype=torch.int32,
                                                device="cpu",
                                                pin_memory=pin_memory)
        self.paged_kv_len_arr_np = (self.paged_kv_len_arr_cpu.numpy())

    def _get_workspace_buffer(self):
        if self._workspace_buffer is None:
            self._workspace_buffer = torch.zeros(
                FLASHINFER_MLA_WORKSPACE_BUFFER_SIZE,
                dtype=torch.uint8,
                device=self.device)
        return self._workspace_buffer

    def _get_decode_wrapper(self,
                            batch_size: int,
                            use_cudagraph: bool = False):
        if use_cudagraph:
            decode_wrapper = self._decode_wrappers_cudagraph.get(
                batch_size, None)
        else:
            decode_wrapper = self._decode_wrapper

        if decode_wrapper is None:
            if use_cudagraph:
                paged_qo_indptr = self.qo_indptr[:batch_size + 1]
                paged_kv_indptr = self.paged_kv_indptr[:batch_size + 1]
                paged_kv_indices = self.paged_kv_indices
                paged_kv_len_arr = self.paged_kv_len_arr[:batch_size]
            else:
                paged_qo_indptr = None
                paged_kv_indptr = None
                paged_kv_indices = None
                paged_kv_len_arr = None
            decode_wrapper = BatchMLAPagedAttentionWrapper(
                self._get_workspace_buffer(),
                use_cuda_graph=use_cudagraph,
                qo_indptr=paged_qo_indptr,
                kv_indptr=paged_kv_indptr,
                kv_indices=paged_kv_indices,
                kv_len_arr=paged_kv_len_arr
                # Tensor cores are enabled by default because the perf would be
                # at least as good as cuda cores for all attention ops in latest
                # gpus.
            )

            # save the decode wrapper
            if use_cudagraph:
                self._decode_wrappers_cudagraph[batch_size] = decode_wrapper
            else:
                self._decode_wrapper = decode_wrapper

        return decode_wrapper

    def _build_decode(self, block_table_tensor: torch.Tensor,
                      seq_lens_cpu: torch.Tensor,
                      seq_lens_device: torch.Tensor,
                      query_start_loc_cpu: torch.Tensor,
                      query_start_loc_device: torch.Tensor,
                      num_decode_tokens: int) -> FlashInferMLAMetadata:
        decode_metadata = FlashInferMLADecodeMetadata(
            block_table=block_table_tensor,
            seq_lens=seq_lens_device,
        )

        paged_kv_indptr_cpu = self.paged_kv_indptr_cpu[:1 + seq_lens_cpu]
        paged_kv_len_arr = self.paged_kv_len_arr[:seq_lens_cpu]
        use_cudagraph = (self.enable_cuda_graph and num_decode_tokens
                         <= self._decode_cudagraph_max_bs)
        if use_cudagraph:
            num_input_tokens = (
                self.vllm_config.pad_for_cudagraph(num_decode_tokens))
            # Carefully fulfill the padding region with reasonable value
            # on cpu.
            # Make sure paged_kv_indptr_cpu is not decreasing
            self.paged_kv_indptr_cpu[1 + num_decode_tokens:1 +
                                     num_input_tokens].fill_(
                                         paged_kv_indptr_cpu[-1])
            # Fill the remaining paged_kv_last_page_len_cpu with 1.
            # This is because flashinfer treats 0 as a full page
            # instead of empty.
            self.paged_kv_last_page_len_cpu[
                num_decode_tokens:num_input_tokens].fill_(1)
        else:
            num_input_tokens = num_decode_tokens

        decode_metadata.decode_wrapper = self._get_decode_wrapper(
            num_input_tokens, use_cudagraph=use_cudagraph)

        decode_metadata.decode_wrapper.plan(
            qo_indptr=self.qo_indptr_cpu,
            kv_indptr=self.paged_kv_indptr_cpu[:num_input_tokens + 1],
            kv_indices=self.paged_kv_indices,
            kv_len_arr=self.paged_kv_len_arr_cpu[:num_input_tokens],
            num_heads=self.num_qo_heads,
            head_dim_ckv=self.num_kv_heads,
            head_dim_kpe=self.mla_dims.qk_rope_head_dim,
            page_size=self.page_size,
            causal=False,
            sm_scale=1.0,  # TODO(Hank) dummy value for testing
            q_data_type=self.q_data_type,
            kv_data_type=self.kv_cache_dtype)


class FlashInferMLAImpl(MLACommonImpl[FlashInferMLAMetadata]):

    def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: int,
            alibi_slopes: Optional[list[float]],
            sliding_window: Optional[int],
            kv_cache_dtype: str,
            logits_soft_cap: Optional[float],
            attn_type: str,
            kv_sharing_target_layer_name: Optional[str],
            # MLA Specific Arguments
            **mla_args) -> None:
        super().__init__(num_heads, head_size, scale, num_kv_heads,
                         alibi_slopes, sliding_window, kv_cache_dtype,
                         logits_soft_cap, attn_type,
                         kv_sharing_target_layer_name, **mla_args)

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashInferMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap")

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "FlashInferMLAImpl")

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "FlashInferMLA V1 with FP8 KV cache not yet supported")

    def _forward_decode(
        self,
        q: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None

        if isinstance(q, tuple):
            q_nope, q_pe = q
            q = torch.cat([q_nope, q_pe], dim=-1)

        # Initialize the MLA wrapper
        mla_wrapper = attn_metadata.decode.decode_wrapper
        head_dim_ckv = q_nope.shape[-1]
        head_dim_kpe = q_pe.shape[-1]

        # Run the MLA computation
        o = mla_wrapper.run(
            q_nope=q_nope,
            q_pe=q_pe,
            ckv_cache=kv_c_and_k_pe_cache[:, :, :head_dim_ckv],
            kpe_cache=kv_c_and_k_pe_cache[:, :, head_dim_ckv:head_dim_ckv +
                                          head_dim_kpe],
            return_lse=False)

        # Return the output tensor and None for LSE (pending support)
        return o, None
