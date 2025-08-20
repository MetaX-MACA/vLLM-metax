# SPDX-License-Identifier: Apache-2.0
from vllm.attention.backends.mla.common import (MLACommonImpl, MLACommonMetadata, MLACommonMetadataBuilder)

from vllm.attention.backends.utils import (PAD_SLOT_ID)

from typing import (TYPE_CHECKING, Any, Dict, Generic, List, Optional, Tuple,
                    Type, TypeVar)

from vllm.model_executor.layers.rotary_embedding import (
    DeepseekScalingRotaryEmbedding, RotaryEmbedding)
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               LinearBase, RowParallelLinear,
                                               UnquantizedLinearMethod)
import torch
from vllm.triton_utils import HAS_TRITON
if HAS_TRITON:
    from vllm.attention.ops.triton_flash_attention import triton_attention
else:
    triton_attention = None

from vllm import envs
from itertools import accumulate
from vllm.attention.backends.utils import PAD_SLOT_ID
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               LinearBase, RowParallelLinear,
                                               UnquantizedLinearMethod)
from vllm.model_executor.layers.rotary_embedding import (
    DeepseekScalingRotaryEmbedding, RotaryEmbedding)
from vllm.triton_utils import HAS_TRITON
from vllm.utils import async_tensor_h2d, cdiv, make_tensor_with_pad, round_down


from vllm import envs

from flash_attn import flash_attn_varlen_func

T = TypeVar("T", bound="MLACommonMetadata")

class MetaxMLACommonMetadataBuilder(MLACommonMetadataBuilder[T], Generic[T]):
    def build(self, seq_lens: List[int], query_lens: List[int],
              cuda_graph_pad_size: int, batch_size: int):
        """Build attention metadata with on-device tensors.

        Args:
            seq_lens: The maybe padded sequence lengths of the input sequences.
            query_lens: The query lengths of the input sequences.
            cuda_graph_pad_size: The padding size for cuda graph.
                                 -1 if cuda graph is not used.
            batch_size: The maybe padded batch size.
        """
        prefix_cache_hit = any([
            inter_data.prefix_cache_hit
            for inter_data in self.input_builder.inter_data_list
        ])

        for inter_data in self.input_builder.inter_data_list:
            self._add_seq_group(inter_data,
                                self.input_builder.chunked_prefill_enabled,
                                prefix_cache_hit)

        device = self.runner.device
        use_captured_graph = cuda_graph_pad_size != -1

        max_query_len = max(query_lens)
        decode_query_lens = query_lens[self.num_prefills:]
        if len(decode_query_lens) > 0:
            max_decode_query_len = max(decode_query_lens)
        else:
            max_decode_query_len = 1
        max_prefill_seq_len = max(self.prefill_seq_lens, default=0)
        max_decode_seq_len = max(self.curr_seq_lens, default=0)
        num_decode_tokens = self.num_decode_tokens
        query_start_loc = list(accumulate(query_lens, initial=0))
        seq_start_loc = list(accumulate(seq_lens, initial=0))

        num_seqs = len(seq_lens)
        if use_captured_graph:
            self.slot_mapping.extend([PAD_SLOT_ID] * cuda_graph_pad_size)
            self.block_tables.extend(self.__class__.BLOCK_TABLE_EXTENDER *
                                     cuda_graph_pad_size)
            num_decode_tokens = batch_size - self.num_prefill_tokens

            block_tables = self._get_graph_runner_block_tables(
                num_seqs, self.block_tables)
        else:
            block_tables = make_tensor_with_pad(
                self.block_tables,
                pad=0,
                dtype=torch.int,
                device=device,
            )
        assert max_query_len > 0, ("query_lens: {}".format(query_lens))

        assert device is not None
        context_lens_tensor = async_tensor_h2d(self.context_lens, torch.int,
                                               device, self.runner.pin_memory)
        seq_lens_tensor = async_tensor_h2d(seq_lens, torch.int, device,
                                           self.runner.pin_memory)

        ############# Metax Modification ############
        # aligned according to batch_size for advance_step_flashattn used
        input_positions_list = self.input_positions
        for _ in range(len(self.input_positions), batch_size):
            input_positions_list.append(0)       
        ############# Metax Modification ############ 

        input_positions = async_tensor_h2d(input_positions_list, torch.long,
                                           device, self.runner.pin_memory)
        slot_mapping_tensor = async_tensor_h2d(self.slot_mapping, torch.long,
                                               device, self.runner.pin_memory)
        query_start_loc_tensor = async_tensor_h2d(query_start_loc, torch.int32,
                                                  device,
                                                  self.runner.pin_memory)
        seq_start_loc_tensor = async_tensor_h2d(seq_start_loc, torch.int32,
                                                device, self.runner.pin_memory)

        context_chunk_cu_seq_lens = None
        context_chunk_starts = None
        context_chunk_seq_tot = None
        context_chunk_max_seq_lens = None

        if (self.chunked_prefill_enabled or self.enable_prefix_caching) \
            and self.num_prefills > 0 \
            and context_lens_tensor is not None \
            and context_lens_tensor[:self.num_prefills].max() > 0:

            # NOTE: it is recommend you read the `Chunked Prefill` section in
            # the comment at the top of the file before trying to understand
            # the following code

            num_prefills_with_context = \
                (context_lens_tensor[:self.num_prefills] > 0).sum().item()

            # currently we allocate an equal amount of workspace for each
            # prefill in the batch, we could probably use a more advanced
            # algorithm here and allocate more workspace to prefills with
            # longer context lengths
            max_context_chunk = \
                self.context_chunk_workspace_size // num_prefills_with_context

            # align max_context_chunk to page_size by rounding down,
            # currently the `gather_cache` kernel cannot handle
            # `context_chunk_starts` that are not aligned to page_size
            max_context_chunk = round_down(max_context_chunk, self.page_size)
            assert max_context_chunk > 0
            num_chunks = cdiv(context_lens_tensor.max(), max_context_chunk)

            # if `max_context_chunk = 256`, `num_chunks = 3`, and
            #   `num_prefills_with_context = 4`, create a tensor that looks like
            #  [[0, 0, 0, 0], [256, 256, 256, 256], [512, 512, 512, 512]]
            context_chunk_starts = \
                torch.arange(num_chunks, device=device, dtype=torch.int32)\
                .unsqueeze(1).expand(-1, self.num_prefills)\
                * max_context_chunk
            chunk_ends = torch.min(context_lens_tensor[:self.num_prefills]\
                .unsqueeze(0), context_chunk_starts + max_context_chunk)
            chunk_seq_lens = (chunk_ends - context_chunk_starts).clamp(min=0)
            _context_chunk_cu_seq_lens = chunk_seq_lens.cumsum(dim=1).to(
                torch.int32)
            zero = torch.zeros(num_chunks, dtype=torch.int32, device=device)\
                .unsqueeze(-1)
            context_chunk_cu_seq_lens = \
                torch.cat([zero, _context_chunk_cu_seq_lens], dim=1)
            context_chunk_max_seq_lens = \
                chunk_seq_lens.max(dim=1).values.tolist()
            context_chunk_seq_tot = chunk_seq_lens.sum(dim=1).tolist()
            assert max(context_chunk_seq_tot) <= \
                self.context_chunk_workspace_size

        return self.runner.attn_backend.make_metadata(
            # Required by ModelRunner
            use_cuda_graph=use_captured_graph,  # Not Attention Related
            # Required by Attention Metadata
            num_prefills=self.num_prefills,
            slot_mapping=slot_mapping_tensor,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            # Required by Attention Metadata (not used)
            multi_modal_placeholder_index_maps=None,  # Not Attention Related
            enable_kv_scales_calculation=False,
            # MLACommonMetadata
            input_positions=input_positions,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=max_query_len,
            max_decode_query_len=max_decode_query_len,
            max_prefill_seq_len=max_prefill_seq_len,
            max_decode_seq_len=max_decode_seq_len,
            query_start_loc=query_start_loc_tensor,
            seq_start_loc=seq_start_loc_tensor,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            head_dim=self.runner.model_config.get_head_size(),
            is_profile_run=self.runner.in_profile_run,
            # MLACommonMetadata Chunk prefill specific
            context_chunk_cu_seq_lens=context_chunk_cu_seq_lens,
            context_chunk_starts=context_chunk_starts,
            context_chunk_seq_tot=context_chunk_seq_tot,
            context_chunk_max_seq_lens=context_chunk_max_seq_lens,
        )

class MetaxMLACommonImpl(MLACommonImpl[T], Generic[T]):
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

        self.rotary_emb = rotary_emb
        self.use_yarn_rope = isinstance(rotary_emb,
                                        DeepseekScalingRotaryEmbedding)
        self.q_proj = q_proj
        self.kv_b_proj = kv_b_proj
        self.o_proj = o_proj

        self.triton_fa_func = triton_attention
        # Handle the differences between the flash_attn_varlen from flash_attn
        # and the one from vllm_flash_attn. The former is used on RoCM and the
        # latter has an additional parameter to control FA2 vs FA3
        """
        self.vllm_flash_attn_version = get_flash_attn_version()
        if self.vllm_flash_attn_version is not None:
            self.flash_attn_varlen_func = \
                functools.partial(flash_attn_varlen_func,
                                  fa_version=self.vllm_flash_attn_version)
        """
        self.flash_attn_varlen_func = flash_attn_varlen_func


        # For MLA the v head dim is smaller than qk head dim so we pad out
        # v with 0s to match the qk head dim for attention backends that do
        # not support different headdims
        # We don't need to pad V if we are on a hopper system with FA3
        """
        self.vllm_flash_attn_version is None or not (
            self.vllm_flash_attn_version == 3
            and current_platform.get_device_capability()[0] == 9)
        """
        self._pad_v = True 

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