# SPDX-License-Identifier: Apache-2.0

# ---------------------------------------------------------------------------
# Note: sample_recovered_tokens_kernel:
#       The original version uses vectorized operations to compute in parallel
#       across the entire vocabulary, whereas the modified Metax version employs
#       chunked processing to handle extremely large vocabularies.
# ---------------------------------------------------------------------------

from vllm.triton_utils import tl, triton

import vllm.v1.sample.rejection_sampler


# NOTE(woosuk): Avoid specialization to prevent unnecessary recompilation.
@triton.jit(do_not_specialize=["max_spec_len"])
def rejection_greedy_sample_kernel(
    output_token_ids_ptr,  # [batch_size, max_spec_len + 1]
    cu_num_draft_tokens_ptr,  # [batch_size]
    draft_token_ids_ptr,  # [num_tokens]
    target_argmax_ptr,  # [num_tokens]
    bonus_token_ids_ptr,  # [batch_size]
    is_greedy_ptr,  # [batch_size] or None
    max_spec_len,
):
    req_idx = tl.program_id(0)
    # FIXME(woosuk): Because is_greedy_ptr is not None at profiling run,
    # re-compilation may happen during runtime when is_greedy_ptr is None.
    is_greedy = True if is_greedy_ptr is None else tl.load(is_greedy_ptr + req_idx)
    # /------------------------  Metax Modification -------------------------\
    if is_greedy is None:
        # Early exit for non-greedy sampling requests.
        return
    # \------------------------- Metax Modification -------------------------/

    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            target_argmax_id = tl.load(target_argmax_ptr + start_idx + pos)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                target_argmax_id,
            )
            if draft_token_id != target_argmax_id:
                # Reject.
                rejected = True

    if not rejected:
        # If all tokens are accepted, append the bonus token.
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )


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
    """Handles large vocabs by chunking to avoid memory constraints."""
    req_idx = tl.program_id(0)
    start_idx = 0 if req_idx == 0 else tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
    end_idx = tl.load(cu_num_draft_tokens_ptr + req_idx)
    num_draft_tokens = end_idx - start_idx

    # Early exit for out-of-range positions.
    pos = tl.program_id(1)
    if pos >= num_draft_tokens:
        return

    max_prob = -float("inf")
    best_token_id = 0

    for block_start in range(0, PADDED_VOCAB_SIZE, BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, vocab_size)

        vocab_offset = tl.arange(0, BLOCK_SIZE)
        if NO_DRAFT_PROBS:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            prob = tl.load(
                target_probs_ptr
                + (start_idx + pos) * vocab_size
                + vocab_offset
                + block_start,
                mask=(
                    (vocab_offset < block_end - block_start)
                    & (vocab_offset + block_start != draft_token_id)
                ),
                other=0,
            )

        else:
            draft_prob = tl.load(
                draft_probs_ptr
                + (start_idx + pos) * vocab_size
                + block_start
                + vocab_offset,
                mask=vocab_offset < block_end - block_start,
                other=0,
            )
            target_prob = tl.load(
                target_probs_ptr
                + (start_idx + pos) * vocab_size
                + vocab_offset
                + block_start,
                mask=vocab_offset < block_end - block_start,
                other=0,
            )
            prob = tl.maximum(target_prob - draft_prob, 0)
            # NOTE(woosuk): We don't need `prob = prob / tl.sum(prob)` here because
            # `tl.argmax` will select the maximum value.

        q = tl.load(
            q_ptr + req_idx * vocab_size + block_start + vocab_offset,
            mask=vocab_offset < block_end - block_start,
            other=float("-inf"),
        )

        # recovered_id = tl.argmax(prob / q, axis=-1)
        # calc block prob and token ID
        block_prob = prob / q
        block_max_prob = tl.max(block_prob, axis=-1)
        block_best_token_id = tl.argmax(block_prob, axis=-1) + block_start

        # update token ID
        max_prob = tl.maximum(max_prob, block_max_prob)
        best_token_id = tl.where(
            block_max_prob >= max_prob, block_best_token_id, best_token_id
        )

    tl.store(output_token_ids_ptr + start_idx + pos, best_token_id)


vllm.v1.sample.rejection_sampler.sample_recovered_tokens_kernel = (
    sample_recovered_tokens_kernel
)

from vllm.v1.sample import rejection_sampler

rejection_sampler.rejection_greedy_sample_kernel = rejection_greedy_sample_kernel
