# SPDX-License-Identifier: Apache-2.0
from vllm.triton_utils import tl, triton

import vllm.v1.sample.rejection_sampler

# SPDX-License-Identifier: Apache-2.0


@triton.jit
def sample_recovered_tokens_kernel(
    output_token_ids_ptr,  # [num_tokens]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    draft_probs_ptr,  # [num_tokens, vocab_size] or None
    target_probs_ptr,  # [num_tokens, vocab_size]
    q_ptr,  # [batch_size, vocab_size]
    vocab_size,
    PADDED_VOCAB_SIZE: tl.constexpr,
    NO_DRAFT_PROBS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr = 1024,
):
    req_idx = tl.program_id(0)
    if req_idx == 0:
        start_idx = 0
    else:
        start_idx = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    # Early exit for out-of-range positions.
    pos = tl.program_id(1)
    if pos >= num_draft_tokens:
        return

    max_prob = -float('inf')
    best_token_id = 0

    for block_start in range(0, PADDED_VOCAB_SIZE, BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, vocab_size)

        vocab_offset = tl.arange(0, BLOCK_SIZE)
        mask = vocab_offset < block_end - block_start

        if NO_DRAFT_PROBS:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            prob = tl.load(
                target_probs_ptr + (start_idx + pos) * vocab_size +
                block_start + vocab_offset,
                mask=(mask & (vocab_offset + block_start != draft_token_id)),
                other=0)

        else:
            draft_prob = tl.load(draft_probs_ptr +
                                 (start_idx + pos) * vocab_size + block_start +
                                 vocab_offset,
                                 mask=mask,
                                 other=0)
            target_prob = tl.load(target_probs_ptr +
                                  (start_idx + pos) * vocab_size +
                                  block_start + vocab_offset,
                                  mask=mask,
                                  other=0)
            prob = tl.maximum(target_prob - draft_prob, 0)

            # NOTE(woosuk): We don't need `prob = prob / tl.sum(prob)` here because
            # `tl.argmax` will select the maximum value.

        q = tl.load(q_ptr + req_idx * vocab_size + block_start + vocab_offset,
                    mask=mask,
                    other=float("-inf"))

        # recovered_id = tl.argmax(prob / q, axis=-1)
        # calc block prob and token ID
        block_prob = prob / q
        block_max_prob = tl.max(block_prob, axis=-1)
        block_best_token_id = tl.argmax(block_prob, axis=-1) + block_start

        # update token ID
        max_prob = tl.maximum(max_prob, block_max_prob)
        best_token_id = tl.where(block_max_prob >= max_prob,
                                 block_best_token_id, best_token_id)

    tl.store(output_token_ids_ptr + start_idx + pos, best_token_id)


vllm.v1.sample.rejection_sampler.sample_recovered_tokens_kernel = sample_recovered_tokens_kernel
