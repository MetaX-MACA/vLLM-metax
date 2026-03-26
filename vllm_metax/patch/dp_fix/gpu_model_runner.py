# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np

from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def _metax_reorder_batch_to_split_decodes_and_prefills(
    input_batch,
    scheduler_output,
    decode_threshold: int = 1,
    group_decodes_by_query_len: bool = False,
) -> bool:
    num_reqs = len(input_batch.req_ids)
    num_scheduled_tokens = [
        scheduler_output.num_scheduled_tokens[req_id] for req_id in input_batch.req_ids
    ]
    num_scheduled_tokens_np = np.array(num_scheduled_tokens, dtype=np.int32)
    num_computed_tokens_np = input_batch.num_computed_tokens_cpu[:num_reqs]

    is_prefill = num_computed_tokens_np == 0
    is_decode = (num_scheduled_tokens_np <= decode_threshold) & (~is_prefill)
    is_extend = (num_scheduled_tokens_np > decode_threshold) & (~is_prefill)

    decode_indices = np.flatnonzero(is_decode)
    if group_decodes_by_query_len and decode_indices.size > 1:
        decode_indices = decode_indices[
            np.argsort(num_scheduled_tokens_np[decode_indices], kind="stable")
        ]
    extend_indices = np.flatnonzero(is_extend)
    prefill_indices = np.flatnonzero(is_prefill)
    target_order = np.concatenate((decode_indices, extend_indices, prefill_indices))

    if np.array_equal(target_order, np.arange(num_reqs, dtype=np.int64)):
        return False

    curr_order = np.arange(num_reqs, dtype=np.int32)
    orig_to_pos = np.arange(num_reqs, dtype=np.int32)

    for dst, src_orig in enumerate(target_order):
        src = int(orig_to_pos[src_orig])
        if src == dst:
            continue

        input_batch.swap_states(dst, src)

        orig_at_dst = int(curr_order[dst])
        curr_order[dst], curr_order[src] = curr_order[src], curr_order[dst]
        orig_to_pos[orig_at_dst], orig_to_pos[src_orig] = src, dst

    return True


class MacaGPUModelRunner(GPUModelRunner):
    def _may_reorder_batch(self, scheduler_output: "SchedulerOutput") -> None:
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return

        if self.reorder_batch_threshold is None:
            return

        spec_decode_enabled = (
            self.speculative_config is not None
            and getattr(self, "num_spec_tokens", 0) > 0
        )
        if spec_decode_enabled:
            group_decodes_by_query_len = getattr(
                self, "_metax_group_decodes_by_query_len", None
            )
            if group_decodes_by_query_len is None:
                group_decodes_by_query_len = any(
                    getattr(
                        group.get_metadata_builder(),
                        "group_decodes_by_query_len",
                        False,
                    )
                    for group in self._attn_group_iterator()
                )
                self._metax_group_decodes_by_query_len = group_decodes_by_query_len
        else:
            group_decodes_by_query_len = False

        _metax_reorder_batch_to_split_decodes_and_prefills(
            self.input_batch,
            scheduler_output,
            decode_threshold=self.reorder_batch_threshold,
            group_decodes_by_query_len=group_decodes_by_query_len,
        )


GPUModelRunner._may_reorder_batch = MacaGPUModelRunner._may_reorder_batch
