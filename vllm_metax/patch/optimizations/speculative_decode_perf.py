# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np

from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.attention.backends.utils import reorder_batch_to_split_decodes_and_prefills


def _metax_reorder_batch_to_split_decodes_and_prefills(
    input_batch,
    scheduler_output,
    decode_threshold: int = 1,
) -> bool:
    """
    Reorder batch to [decode (sorted), extend, prefill] for query length bucketing.

    Key feature: Sorts decode requests by scheduled token count (ascending).
    This enables FlashAttention query length bucketing by grouping similar requests.

    Request classification:
    - Prefill: computed_tokens == 0 (new requests, compute-bound)
    - Decode: scheduled_tokens <= threshold (ongoing, memory-bound, typically 1 token)
    - Extend: scheduled_tokens > threshold (ongoing, compute-bound, multiple tokens)

    Args:
        input_batch: Batch containing request states and current ordering
        scheduler_output: Scheduler output with num_scheduled_tokens per request id
        decode_threshold: Maximum scheduled tokens to classify as decode (default: 1)

    Returns:
        True if batch was reordered, False if already in target order

    Example (threshold=3):
        reqs: [ext:4, pre:4, dec:2, dec:1, ext:5, pre:1, dec:3, pre:2]
        result: [dec:1, dec:2, dec:3, ext:4, ext:5, pre:4, pre:1, pre:2]
                (decodes sorted by token count, then extends, then prefills)
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

    # Classify requests by type
    is_prefill = num_computed_tokens_np == 0
    is_decode = (num_scheduled_tokens_np <= decode_threshold) & (~is_prefill)
    is_extend = (num_scheduled_tokens_np > decode_threshold) & (~is_prefill)

    # Extract indices for each type, sorting decode requests by token count
    decode_indices = np.flatnonzero(is_decode)
    if decode_indices.size > 1:
        decode_indices = decode_indices[
            np.argsort(num_scheduled_tokens_np[decode_indices], kind="stable")
        ]

    extend_indices = np.flatnonzero(is_extend)
    prefill_indices = np.flatnonzero(is_prefill)

    # Build target order: decode → extend → prefill
    target_order = np.concatenate([decode_indices, extend_indices, prefill_indices])

    # Early exit if no reordering needed
    if np.array_equal(target_order, np.arange(num_reqs, dtype=np.int32)):
        return False

    # Track current order and request positions for efficient swapping
    curr_order = np.arange(num_reqs, dtype=np.int32)
    orig_to_pos = np.arange(num_reqs, dtype=np.int32)

    # Perform reordering via element-wise swaps
    for target_pos, src_orig in enumerate(target_order):
        src = int(orig_to_pos[src_orig])
        if src == target_pos:
            continue

        input_batch.swap_states(src, target_pos)

        # Update tracking after swap
        orig_at_target = int(curr_order[target_pos])
        curr_order[target_pos], curr_order[src] = (
            curr_order[src],
            curr_order[target_pos],
        )
        orig_to_pos[orig_at_target], orig_to_pos[src_orig] = src, target_pos

    return True


class MacaGPUModelRunner(GPUModelRunner):
    def _may_reorder_batch(self, scheduler_output: "SchedulerOutput") -> None:
        if (
            len(self.kv_cache_config.kv_cache_groups) == 0
            or self.reorder_batch_threshold is None
        ):
            return

        # Determine if decode grouping should be enabled
        use_decode_grouping = False
        if (
            self.speculative_config is not None
            and self.speculative_config.num_speculative_tokens > 0
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
            reorder_batch_to_split_decodes_and_prefills(
                self.input_batch,
                scheduler_output,
                decode_threshold=self.reorder_batch_threshold,
            )


GPUModelRunner._may_reorder_batch = MacaGPUModelRunner._may_reorder_batch
