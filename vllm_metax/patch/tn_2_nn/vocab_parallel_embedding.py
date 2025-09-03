# SPDX-License-Identifier: Apache-2.0

import vllm

from vllm.logger import init_logger

logger = init_logger(__name__)

import torch
from typing import Tuple
from vllm_metax.model_executor.layers.vocab_parallel_embedding import (UnquantizedEmbeddingMethod, 
    VocabParallelEmbedding,
    ParallelLMHead)

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
 
vllm.model_executor.layers.vocab_parallel_embedding.UnquantizedEmbeddingMethod = UnquantizedEmbeddingMethod
vllm.model_executor.layers.vocab_parallel_embedding.get_masked_input_and_mask = torch.jit.script(get_masked_input_and_mask)
vllm.model_executor.layers.vocab_parallel_embedding.VocabParallelEmbedding = VocabParallelEmbedding
vllm.model_executor.layers.vocab_parallel_embedding.ParallelLMHead = ParallelLMHead






