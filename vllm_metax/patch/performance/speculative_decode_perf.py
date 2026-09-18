# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# -----------------------------------------------------------------------------
# Note: Optimize speculative decode batch reordering for MetaX query-length
#       bucketing.
#
# Affected versions: vLLM 0.29.1.dev0 (98dff2a81d), verified 2026-09-17.
#
# Remove at: Upstream batch reordering supports decode query-length buckets required by
#     the MetaX attention backend.
# -----------------------------------------------------------------------------

import numpy as np

from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from typing import TYPE_CHECKING
from vllm_metax.patch.utils import patch

_original_may_reorder_batch = GPUModelRunner._may_reorder_batch

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

# ---------------------------------
# Note:
# _metax_reorder_batch_to_split_decodes_and_prefills provides MetaX-specific
# batch reordering for speculative decode. It preserves upstream's four request
# regions and additionally sorts decodes by scheduled token count. Equal query
# lengths then form contiguous buckets for the MetaX FlashAttention backend.


def _metax_reorder_batch_to_split_decodes_and_prefills(
    input_batch,
    scheduler_output,
    decode_threshold: int = 1,
) -> bool:
    """Reorder requests into four regions and bucket decodes by query length.

    Target order:
        decode (sorted) -> short_extend -> long_extend -> prefill

    The scheduled token count is this step's query length, not the request's
    total sequence length. Speculative verification can schedule different
    numbers of tokens for different decode requests. Sorting decodes by this
    count puts equal query lengths next to each other, so FlashAttention can
    process them as contiguous buckets.

    Request classification (mutually exclusive):
        - Prefill: computed_tokens == 0; the request has no computed context.
          This remains a first prefill even when its query length is short.
        - Decode: computed_tokens > 0, the prompt is fully computed, and
          scheduled_tokens <= decode_threshold.
        - Short extend: computed_tokens > 0, the prompt is not fully computed,
          and scheduled_tokens <= decode_threshold.
        - Long extend: computed_tokens > 0 and scheduled_tokens > threshold.
          As in upstream, this category does not require an unfinished prompt.

    Why separate short and long extends?
        Upstream split_decodes_and_prefills expects these four regions. A
        backend may include short extends in its decode path, or treat them
        as prefills via treat_short_extends_as_decodes=False. Keeping short
        extends between true decodes and long extends supports both choices
        with a contiguous split. Mixing short extends into the sorted decodes
        could put an unfinished prompt before a true decode and move that
        backend's prefill boundary too early.

    Ordering within each region:
        Only decodes are sorted, in ascending query length. The sort is stable
        for equal lengths. Short extends, long extends and first prefills each
        retain their original relative order.

    Differences from upstream reorder_batch_to_split_decodes_and_prefills:
        - Classification and region order are the same. Upstream does not sort
          decodes by query length; MetaX adds that ordering for decode buckets.
        - Upstream only moves requests occupying the wrong region. Its stable
          sort applies to that subset, so it does not guarantee stable order
          across the whole region. MetaX preserves the relative order of all
          non-decode requests within their respective regions, and of decodes
          with equal query lengths.
        - Upstream builds a source/destination map for misplaced requests and
          resolves it through swaps. MetaX builds a complete target permutation
          and tracks both current positions and original request indices while
          placing requests from left to right.
        - Upstream returns False once all requests occupy the correct regions.
          MetaX returns False only when the complete target order already
          matches, including decode query-length ordering.

        For example, with threshold=3, the following batch already has correct
        regions, so upstream leaves it unchanged:
            upstream: [D:3, D:1, D:2, D:1, S:2, L:8, P:4]
            MetaX:    [D:1, D:1, D:2, D:3, S:2, L:8, P:4]
        Equal-length decodes form contiguous buckets after the MetaX reorder.
        The additional sorting and potentially more swaps add CPU overhead;
        the net performance benefit depends on the batch and attention backend.

    Example with decode_threshold=3 (D=decode, S=short extend, L=long extend,
    P=first prefill; the number after ':' is the scheduled token count):
        before: [L:4, P:4, D:2, S:2, D:1, L:5, P:1, D:3, S:1, P:2]
        after:  [D:1, D:2, D:3, S:2, S:1, L:4, L:5, P:4, P:1, P:2]
        Notice that S:2 stays before S:1, and P:1 stays in the prefill region.

    Args:
        input_batch: Batch whose request states are reordered in place through
            swap_states, keeping per-request data aligned with req_ids.
        scheduler_output: Supplies num_scheduled_tokens for each request ID.
        decode_threshold: Maximum query length eligible for the decode or
            short-extend region. Defaults to 1.

    Returns:
        True if request states were swapped; False if already in target order.
    """
    num_reqs = len(input_batch.req_ids)

    # Collect scheduled and computed tokens for all requests
    num_scheduled_tokens_np = np.array(
        [
            scheduler_output.num_scheduled_tokens[req_id]
            for req_id in input_batch.req_ids
        ],
        dtype=np.int32,
    )
    num_computed_tokens_np = input_batch.num_computed_tokens_cpu[:num_reqs]

    # A short query is not necessarily a decode: check prompt completion too.
    is_prefill = num_computed_tokens_np == 0
    done_prefilling = num_computed_tokens_np >= input_batch.num_prompt_tokens[:num_reqs]
    is_below_threshold = num_scheduled_tokens_np <= decode_threshold
    is_decode = is_below_threshold & (~is_prefill) & done_prefilling
    is_short_extend = is_below_threshold & (~is_prefill) & (~done_prefilling)
    is_long_extend = (~is_below_threshold) & (~is_prefill)

    # Stable sorting keeps equal-length decodes together without changing their
    # relative order. flatnonzero preserves input order for the other regions.
    decode_indices = np.flatnonzero(is_decode)
    if decode_indices.size > 1:
        decode_indices = decode_indices[
            np.argsort(num_scheduled_tokens_np[decode_indices], kind="stable")
        ]

    short_extend_indices = np.flatnonzero(is_short_extend)
    long_extend_indices = np.flatnonzero(is_long_extend)
    prefill_indices = np.flatnonzero(is_prefill)

    # /-------------------- MetaX Modification --------------------\
    # Preserve upstream's regions while bucketing decodes by query length.
    target_order = np.concatenate(
        [decode_indices, short_extend_indices, long_extend_indices, prefill_indices]
    )
    # \-------------------- MetaX Modification --------------------/

    # Early exit if no reordering needed
    if np.array_equal(target_order, np.arange(num_reqs, dtype=np.int32)):
        return False

    # target_order contains ORIGINAL request indices, but swap_states changes
    # their positions. Keep both directions so each next request can be found
    # without searching the batch:
    #   curr_order[position] = original request index currently at that position
    #   orig_to_pos[original request index] = its current position
    curr_order = np.arange(num_reqs, dtype=np.int32)
    orig_to_pos = np.arange(num_reqs, dtype=np.int32)

    # Fill the target order from left to right. Each swap places one request in
    # its final position; positions before target_pos are already correct.
    for target_pos, src_orig in enumerate(target_order):
        src = int(orig_to_pos[src_orig])
        if src == target_pos:
            continue

        input_batch.swap_states(src, target_pos)

        # Update both maps for the placed request and the displaced request.
        orig_at_target = int(curr_order[target_pos])
        curr_order[target_pos], curr_order[src] = (
            curr_order[src],
            curr_order[target_pos],
        )
        orig_to_pos[orig_at_target], orig_to_pos[src_orig] = src, target_pos

    return True


@patch(
    "vllm.v1.worker.gpu_model_runner",
    "GPUModelRunner._may_reorder_batch",
)
def _may_reorder_batch(self, scheduler_output: "SchedulerOutput") -> None:
    if (
        len(self.kv_cache_config.kv_cache_groups) == 0
        or self.reorder_batch_threshold is None
    ):
        return

    # Use the MetaX reorder only when speculative decoding is enabled and at
    # least one attention metadata builder requests query-length grouping.
    # Otherwise, delegate to upstream's ordinary batch-reordering method.
    use_decode_grouping = False
    if (
        self.speculative_config is not None
        and (self.speculative_config.num_speculative_tokens or 0) > 0
    ):
        if not hasattr(self, "_use_decode_grouping"):
            self._use_decode_grouping = any(
                getattr(
                    group.get_metadata_builder(),
                    "group_decodes_by_query_len",
                    False,
                )
                for group in self._attn_group_iterator()
            )
        use_decode_grouping = self._use_decode_grouping

    # Call appropriate reorder function
    if use_decode_grouping:
        _metax_reorder_batch_to_split_decodes_and_prefills(
            self.input_batch,
            scheduler_output,
            decode_threshold=self.reorder_batch_threshold,
        )
    else:
        _original_may_reorder_batch(self, scheduler_output)
