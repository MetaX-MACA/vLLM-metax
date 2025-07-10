# SPDX-License-Identifier: Apache-2.0

import functools
from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Optional, TypeVar

import torch

from vllm.attention.backends.utils import get_mla_dims
from vllm.attention.ops.merge_attn_states import merge_attn_states
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               LinearBase, RowParallelLinear,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
from vllm.platforms import current_platform
from vllm.utils import cdiv, round_down

from vllm.v1.attention.backends.mla.common import (MLACommonDecodeMetadata,
                                                   MLACommonBackend,
                                                   MLACommonPrefillMetadata,
                                                   MLACommonMetadata,
                                                   MLACommonMetadataBuilder,
                                                   MLACommonImpl)
# For rocm use upstream flash attention
from flash_attn import flash_attn_varlen_func
is_vllm_fa = False

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    
from vllm import envs

logger = init_logger(__name__)

def get_flash_attn_version():
    return None


class MetaxMLACommonBackend(MLACommonBackend):

    @staticmethod
    def get_builder_cls() -> type["MLACommonMetadataBuilder"]:
        return MetaxMLACommonMetadataBuilder


D = TypeVar("D", bound=MLACommonDecodeMetadata)
M = TypeVar("M", bound=MLACommonMetadata)


class MetaxMLACommonMetadataBuilder(MLACommonMetadataBuilder[M], Generic[M]):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

    def __init__(self,
                 runner: "GPUModelRunner",
                 metadata_cls: Optional[type[M]] = None):
        self.metadata_cls = metadata_cls \
            if metadata_cls is not None else MLACommonMetadata
        self.runner = runner
        scheduler_config = runner.scheduler_config
        model_config = runner.model_config
        cache_config = runner.cache_config
        self.chunked_prefill_enabled = scheduler_config.chunked_prefill_enabled
        self.num_heads = model_config.get_num_attention_heads(
            runner.parallel_config)
        self.mla_dims = get_mla_dims(model_config)
        self.aot_schedule = False

        # Dont try to access the runner on AMD
        if self.aot_schedule:
            self.page_size = self.runner.block_size

        if self.chunked_prefill_enabled:
            self.chunked_prefill_workspace_size = min(
                # Max sure there is enough for 8 full length request or at least
                # 4 pages of cache per request
                max(
                    8 * model_config.max_model_len, 4 *
                    scheduler_config.max_num_seqs * cache_config.block_size),
                # For long-context models try not to over-allocate limiting
                # kv-cache space, limiting it to 64k tokens,
                # which would result in the workspace being:
                #   2*(576)*(64*1024) = 144mb
                # (assuming 576 MLA head dim, and fp16)
                # which would result in up-projected context being
                #   2*(192*128)*(64*1024) = 3gb
                # (assuming 192 QK head dim, 128 heads, and fp16)
                128 * 1024)
            assert self.chunked_prefill_workspace_size >= \
                scheduler_config.max_num_seqs * cache_config.block_size
            self.chunked_prefill_workspace = torch.empty(
                (self.chunked_prefill_workspace_size,
                 model_config.get_head_size()),
                dtype=model_config.dtype,
                device=runner.device,
            )

    def build(self, num_reqs: int, num_actual_tokens: int, max_query_len: int,
              common_prefix_len: int) -> M:
        assert self._num_decodes + self._num_prefills == num_reqs

        # Note(simon): be careful about the CPU <> GPU memory movement in this
        # function. We should avoid GPU -> CPU sync as much as possible because
        # it blocks on all previous kernels.
        device = self.runner.device
        block_table = (
            self.runner.input_batch.block_table.get_device_tensor()[:num_reqs])
        query_start_loc = self.runner.query_start_loc_cpu[:num_reqs + 1].to(
            device, non_blocking=True)
        slot_mapping = self.runner.slot_mapping_cpu[:num_actual_tokens].to(
            device, non_blocking=True).long()
        input_positions = self.runner.positions_cpu[:num_actual_tokens].to(
            device, non_blocking=True).long()

        seq_lens_cpu = self.runner.seq_lens_cpu[:num_reqs]
        seq_lens = seq_lens_cpu.to(device, non_blocking=True)

        prefill_metadata = None
        if self._num_prefills > 0:
            reqs_start = self._num_decodes  # prefill_start
            tokens_start = self._num_decode_tokens

            context_lens_cpu = self.runner.input_batch.\
                num_computed_tokens_cpu_tensor[reqs_start:num_reqs]
            max_context_len_cpu = context_lens_cpu.max().item()
            num_prefills_with_context_cpu = (context_lens_cpu > 0).sum().item()
            prefill_query_start_loc = query_start_loc[
                reqs_start:] - query_start_loc[reqs_start]

            chunked_context_metadata = None
            if self.chunked_prefill_enabled and self._num_prefills > 0 \
                and max_context_len_cpu > 0:
                # NOTE: it is recommend you read the `Chunked Prefill` section
                # in the comment at the top of the file before trying to
                # understand the following code

                # currently we allocate an equal amount of workspace for each
                # prefill in the batch, we could probably use a more advanced
                # algorithm here and allocate more workspace to prefills with
                # longer context lengths
                max_context_chunk = (self.chunked_prefill_workspace_size //
                                     num_prefills_with_context_cpu)

                # Metax Modification
                if self.aot_schedule:
                    # align max_context_chunk to page_size by rounding down,
                    # currently the `gather_cache` kernel cannot handle
                    # `context_chunk_starts` that are not aligned to page_size
                    max_context_chunk = round_down(max_context_chunk,
                                                   self.page_size)

                assert max_context_chunk > 0
                num_chunks = cdiv(max_context_len_cpu, max_context_chunk)

                # if `max_context_chunk = 256`, `num_chunks = 3`, and
                #   `num_prefills_with_context = 4`, create a tensor that looks
                # like
                #  [[0, 0, 0, 0], [256, 256, 256, 256], [512, 512, 512, 512]]
                # Note(simon): this is done in CPU because of downstream's
                # of `to_list`.
                chunk_starts = \
                    torch.arange(num_chunks, dtype=torch.int32) \
                    .unsqueeze(1).expand(-1, self._num_prefills) \
                    * max_context_chunk
                chunk_ends = torch.min(context_lens_cpu.unsqueeze(0),
                                       chunk_starts + max_context_chunk)
                chunk_seq_lens = (chunk_ends - chunk_starts).clamp(min=0)

                cu_seq_lens_cpu = torch.zeros(num_chunks,
                                              self._num_prefills + 1,
                                              dtype=torch.int32,
                                              pin_memory=True)
                torch.cumsum(chunk_seq_lens,
                             dim=1,
                             out=cu_seq_lens_cpu[:, 1:],
                             dtype=torch.int32)

                chunked_context_metadata = \
                    MLACommonPrefillMetadata.ChunkedContextMetadata(
                    cu_seq_lens=cu_seq_lens_cpu.to(device, non_blocking=True),
                    starts=chunk_starts.to(device, non_blocking=True),
                    seq_tot=chunk_seq_lens.sum(dim=1).tolist(),
                    max_seq_lens=chunk_seq_lens.max(dim=1).values.tolist(),
                    workspace=self.chunked_prefill_workspace,
                )

                assert max(chunked_context_metadata.max_seq_lens) <= \
                    self.chunked_prefill_workspace_size

            prefill_metadata = MLACommonPrefillMetadata(
                input_positions=input_positions[tokens_start:],
                block_table=block_table[reqs_start:, ...],
                query_start_loc=prefill_query_start_loc,
                max_query_len=max_query_len,
                chunked_context=chunked_context_metadata,
            )

        decode_metadata = None
        if self._num_decodes > 0:
            decode_metadata = self._build_decode(
                input_positions=input_positions[:self._num_decode_tokens],
                block_table=block_table[:self._num_decodes, ...],
                seq_lens=seq_lens[:self._num_decodes],
            )

        return self.metadata_cls(
            num_actual_tokens=num_actual_tokens,
            query_start_loc=query_start_loc,
            slot_mapping=slot_mapping,
            head_dim=self.runner.model_config.get_head_size(),
            # MLACommonMetadata Chunk prefill specific
            num_decodes=self._num_decodes,
            num_decode_tokens=self._num_decode_tokens,
            num_prefills=self._num_prefills,
            prefill=prefill_metadata,
            decode=decode_metadata,
        )



class MetaxMLACommonImpl(MLACommonImpl[M], Generic[M]):
    """
    NOTE: Please read the comment at the top of the file before trying to
    understand this class
    """

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
        q_lora_rank: Optional[int],
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        rotary_emb: RotaryEmbedding,
        # q_proj should be q_b_proj if q_lora_rank is not None, but from an
        # attention backend perspective we rely on the layer to pass in the
        # correct matrix
        q_proj: ColumnParallelLinear,
        kv_b_proj: ColumnParallelLinear,
        o_proj: RowParallelLinear,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim

        # Hack for V1 for now to avoid torch library overhead (since we are
        # already inside an attention custom op), pull out the forward
        # method from the rotary embedding and call it directly
        # TODO(lucas): we should probably find a cleaner way to do this
        self.rotary_emb = rotary_emb.forward_native
        if current_platform.is_cuda():
            self.rotary_emb = rotary_emb.forward_cuda

        self.q_proj = q_proj
        self.kv_b_proj = kv_b_proj
        self.o_proj = o_proj

        # Handle the differences between the flash_attn_varlen from flash_attn
        # and the one from vllm_flash_attn. The former is used on RoCM and the
        # latter has an additional parameter to control FA2 vs FA3
        self.flash_attn_varlen_func = flash_attn_varlen_func
        self.vllm_flash_attn_version = None


        # For MLA the v head dim is smaller than qk head dim so we pad out
        # v with 0s to match the qk head dim for attention backends that do
        # not support different headdims
        # We don't need to pad V if we are on a hopper system with FA3
        self._pad_v = True

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
            return_attn_probs=return_softmax_lse,
            softmax_scale=softmax_scale,
            **kwargs,
        )

        # Unpack the output if there is multiple results
        lse = None
        if isinstance(attn_out, tuple):
            attn_out, lse = attn_out[0], attn_out[1]

        # unpad if necessary
        if self._pad_v:
            attn_out = attn_out.view(-1, self.num_heads, q.shape[-1])[..., :v.shape[-1]]\
                .reshape(-1, self.num_heads * v.shape[-1])

        # Remain consistent with old `flash_attn_varlen_func` where there
        # is only one output tensor if `return_softmax_lse` is False.
        if return_softmax_lse:
            return attn_out, lse
        return attn_out

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

    def _forward_prefill(
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: MLACommonMetadata,
    ) -> torch.Tensor:
        assert attn_metadata.prefill is not None

        has_context = attn_metadata.prefill.chunked_context is not None
        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(\
            -1, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = kv_nope\
            .split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k = torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)

        output = self._flash_attn_varlen_diff_headdims(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=attn_metadata.prefill.query_start_loc,
            cu_seqlens_k=attn_metadata.prefill.query_start_loc,
            max_seqlen_q=attn_metadata.prefill.max_query_len,
            max_seqlen_k=attn_metadata.prefill.max_query_len,
            softmax_scale=self.scale,
            causal=True,
            # Metax Modification
            # return_softmax_lse=has_context,
        )

        if False and has_context:
            suffix_output, suffix_lse = output
            context_output, context_lse = self._compute_prefill_context( \
                q, kv_c_and_k_pe_cache, attn_metadata)

            output = torch.empty_like(suffix_output)
            merge_attn_states(
                output=output,
                prefix_output=context_output,
                prefix_lse=context_lse,
                suffix_output=suffix_output,
                suffix_lse=suffix_lse,
            )

        return self.o_proj(output)[0]
