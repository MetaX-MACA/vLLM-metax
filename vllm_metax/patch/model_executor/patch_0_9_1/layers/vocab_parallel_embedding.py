# SPDX-License-Identifier: Apache-2.0

import vllm
from vllm_metax.patch.hook_registry import register_patch
from vllm.logger import init_logger

logger = init_logger(__name__)

import torch
import inspect
from vllm import envs
from typing import Optional, Tuple
from vllm.platforms import current_platform
from torch.nn.parameter import Parameter, UninitializedParameter
from vllm_metax.model_executor.layers.vocab_parallel_embedding import (UnquantizedEmbeddingMethod, 
    VocabParallelEmbedding,
    ParallelLMHead)

DEFAULT_VOCAB_PADDING_SIZE = 64

def get_masked_input_and_mask(
        input_: torch.Tensor, org_vocab_start_index: int,
        org_vocab_end_index: int, num_org_vocab_padding: int,
        added_vocab_start_index: int,
        added_vocab_end_index: int) -> Tuple[torch.Tensor, torch.Tensor]:
    # torch.compile will fuse all of the pointwise ops below
    # torch.jit.script will fuse all of the pointwise ops below
    # into a single kernel, making it very fast
    org_vocab_mask = (input_ >= org_vocab_start_index) & (
        input_ < org_vocab_end_index)
    added_vocab_mask = (input_ >= added_vocab_start_index) & (
        input_ < added_vocab_end_index)
    added_offset = added_vocab_start_index - (
        org_vocab_end_index - org_vocab_start_index) - num_org_vocab_padding
    valid_offset = (org_vocab_start_index *
                    org_vocab_mask) + (added_offset * added_vocab_mask)
    vocab_mask = org_vocab_mask | added_vocab_mask
    input_ = vocab_mask * (input_ - valid_offset)
    return input_, ~vocab_mask

def vocab_enbedding_weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
    output_dim = getattr(param, "output_dim", None)
    packed_dim = getattr(param, "packed_dim", None)

    # If the parameter is a gguf weight, then load it directly.
    if getattr(param, "is_gguf_weight_type", None):
        param.data.copy_(loaded_weight)
        param.weight_type = loaded_weight.item()
        return
    elif isinstance(param, UninitializedParameter):
        shape = list(loaded_weight.shape)
        if output_dim is not None:
            shape[output_dim] = self.num_embeddings_per_partition
        param.materialize(tuple(shape), dtype=loaded_weight.dtype)

    # If parameter does not have output dim, then it should
    # be copied onto all gpus (e.g. g_idx for act_order gptq).
    if output_dim is None:
        assert param.data.shape == loaded_weight.shape
        param.data.copy_(loaded_weight)
        return

    # Shard indexes for loading the weight
    start_idx = self.shard_indices.org_vocab_start_index
    shard_size = self.shard_indices.org_vocab_end_index - start_idx

    # If param packed on the same dim we are sharding on, then
    # need to adjust offsets of loaded weight by pack_factor.
    if packed_dim is not None and packed_dim == output_dim:
        packed_factor = param.packed_factor if isinstance(
            param, BasevLLMParameter) else param.pack_factor
        assert loaded_weight.shape[output_dim] == (self.org_vocab_size //
                                                    param.packed_factor)
        start_idx = start_idx // packed_factor
        shard_size = shard_size // packed_factor
    else:
        assert loaded_weight.shape[output_dim] == self.org_vocab_size

    # Copy the data. Select chunk corresponding to current shard.
    loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)

    if envs.MACA_VLLM_USE_TN_2_NN:
        loaded_weight = loaded_weight.t()
        # we should padding last dimension after weight transpose
        padding_needed = max(self.num_embeddings_per_partition - loaded_weight.size(-1), 0)
        if padding_needed:
            loaded_weight = torch.nn.functional.pad(loaded_weight, (0, padding_needed), value=0)

    if current_platform.is_hpu():
        # FIXME(kzawora): Weight copy with slicing bugs out on Gaudi here,
        # so we're using a workaround. Remove this when fixed in
        # HPU PT bridge.
        padded_weight = torch.cat([
            loaded_weight,
            torch.zeros(param.shape[0] - loaded_weight.shape[0],
                        *loaded_weight.shape[1:])
        ])
        param.data.copy_(padded_weight)
    else:
        param[:loaded_weight.shape[0]].data.copy_(loaded_weight)
        param[loaded_weight.shape[0]:].data.fill_(0)
        
vllm.model_executor.layers.vocab_parallel_embedding.UnquantizedEmbeddingMethod = UnquantizedEmbeddingMethod
vllm.model_executor.layers.vocab_parallel_embedding.get_masked_input_and_mask = torch.jit.script(get_masked_input_and_mask)
vllm.model_executor.layers.vocab_parallel_embedding.VocabParallelEmbedding = VocabParallelEmbedding
vllm.model_executor.layers.vocab_parallel_embedding.ParallelLMHead = ParallelLMHead

register_patch("vllm.model_executor.layers.vocab_parallel_embedding", "UnquantizedEmbeddingMethod", UnquantizedEmbeddingMethod)
register_patch("vllm.model_executor.layers.vocab_parallel_embedding", "get_masked_input_and_mask", get_masked_input_and_mask)
register_patch("vllm.model_executor.layers.vocab_parallel_embedding", "VocabParallelEmbedding", VocabParallelEmbedding)
register_patch("vllm.model_executor.layers.vocab_parallel_embedding", "ParallelLMHead", ParallelLMHead)

