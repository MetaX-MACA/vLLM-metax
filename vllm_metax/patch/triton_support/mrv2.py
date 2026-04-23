# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.


# ---------------------------------------------------------
# TODO(hank): remove this once triton3.6+metax is released
#     triton 3.0 does not support _load_ptr on compile time
# --------------------------------------------------------
from vllm.triton_utils import tl, triton


@triton.jit
def _load_ptr_i32(ptr_to_ptr):
    ptr = tl.load(ptr_to_ptr)
    return tl.cast(ptr, tl.pointer_type(tl.int32))


@triton.jit(do_not_specialize=["num_reqs"])
def _gather_block_tables_kernel(
    batch_idx_to_req_idx,  # [batch_size]
    src_block_table_ptrs,  # [num_kv_cache_groups]
    dst_block_table_ptrs,  # [num_kv_cache_groups]
    block_table_strides,  # [num_kv_cache_groups]
    num_blocks_ptr,  # [num_kv_cache_groups, max_num_reqs]
    num_blocks_stride,
    num_reqs,  # actual number of requests (for padding)
    max_num_blocks,  # stride for zeroing padded rows
    BLOCK_SIZE: tl.constexpr,
):
    # kv cache group id
    group_id = tl.program_id(0)
    batch_idx = tl.program_id(1)

    stride = tl.load(block_table_strides + group_id)
    dst_block_table_ptr = _load_ptr_i32(dst_block_table_ptrs + group_id)
    dst_row_ptr = dst_block_table_ptr + batch_idx * stride

    if batch_idx >= num_reqs:
        # Zero out padded rows.
        for i in tl.range(0, max_num_blocks, BLOCK_SIZE):
            offset = i + tl.arange(0, BLOCK_SIZE)
            tl.store(dst_row_ptr + offset, 0, mask=offset < max_num_blocks)
        return

    req_idx = tl.load(batch_idx_to_req_idx + batch_idx)
    group_num_blocks_ptr = num_blocks_ptr + group_id * num_blocks_stride
    num_blocks = tl.load(group_num_blocks_ptr + req_idx)

    src_block_table_ptr = _load_ptr_i32(src_block_table_ptrs + group_id)
    src_row_ptr = src_block_table_ptr + req_idx * stride

    for i in tl.range(0, num_blocks, BLOCK_SIZE):
        offset = i + tl.arange(0, BLOCK_SIZE)
        block_ids = tl.load(src_row_ptr + offset, mask=offset < num_blocks)
        tl.store(dst_row_ptr + offset, block_ids, mask=offset < num_blocks)


@triton.jit
def _compute_slot_mappings_kernel(
    max_num_tokens,
    idx_mapping,  # [num_reqs]
    query_start_loc,  # [num_reqs + 1]
    pos,  # [num_tokens]
    block_table_ptrs,  # [num_kv_cache_groups]
    block_table_strides,  # [num_kv_cache_groups]
    block_sizes,  # [num_kv_cache_groups]
    slot_mappings_ptr,  # [num_kv_cache_groups, max_num_tokens]
    slot_mappings_stride,
    cp_rank,
    CP_SIZE: tl.constexpr,
    CP_INTERLEAVE: tl.constexpr,
    PAD_ID: tl.constexpr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    # kv cache group id
    group_id = tl.program_id(0)
    batch_idx = tl.program_id(1)
    slot_mapping_ptr = slot_mappings_ptr + group_id * slot_mappings_stride

    if batch_idx == tl.num_programs(1) - 1:
        # Pad remaining slots to -1. This is needed for CUDA graphs.
        # Start from actual token count (not padded) to cover the gap
        # between actual tokens and padded tokens that can contain stale
        # valid slot IDs from previous chunks during chunked prefill.
        actual_num_tokens = tl.load(query_start_loc + batch_idx)
        for i in range(actual_num_tokens, max_num_tokens, TRITON_BLOCK_SIZE):
            offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
            tl.store(slot_mapping_ptr + offset, PAD_ID, mask=offset < max_num_tokens)
        return

    block_table_ptr = _load_ptr_i32(block_table_ptrs + group_id)
    block_table_stride = tl.load(block_table_strides + group_id)
    block_size = tl.load(block_sizes + group_id)

    req_state_idx = tl.load(idx_mapping + batch_idx)
    start_idx = tl.load(query_start_loc + batch_idx)
    end_idx = tl.load(query_start_loc + batch_idx + 1)
    for i in range(start_idx, end_idx, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        positions = tl.load(pos + offset, mask=offset < end_idx, other=0)

        block_indices = positions // (block_size * CP_SIZE)
        block_offsets = positions % (block_size * CP_SIZE)
        block_numbers = tl.load(
            block_table_ptr + req_state_idx * block_table_stride + block_indices
        )

        if CP_SIZE == 1:
            # Common case: Context parallelism is not used.
            slot_ids = block_numbers * block_size + block_offsets
        else:
            # Context parallelism is used.
            is_local = block_offsets // CP_INTERLEAVE % CP_SIZE == cp_rank
            rounds = block_offsets // (CP_INTERLEAVE * CP_SIZE)
            remainder = block_offsets % CP_INTERLEAVE
            local_offsets = rounds * CP_INTERLEAVE + remainder
            slot_ids = block_numbers * block_size + local_offsets
            slot_ids = tl.where(is_local, slot_ids, PAD_ID)

        tl.store(slot_mapping_ptr + offset, slot_ids, mask=offset < end_idx)


@triton.jit
def _penalties_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    token_ids_ptr,
    expanded_local_pos_ptr,
    repetition_penalty_ptr,
    frequency_penalty_ptr,
    presence_penalty_ptr,
    prompt_bin_mask_ptr,
    prompt_bin_mask_stride,
    output_bin_counts_ptr,
    output_bin_counts_stride,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    MAX_SPEC_LEN: tl.constexpr,
):
    token_idx = tl.program_id(0)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    rep_penalty = tl.load(repetition_penalty_ptr + req_state_idx)
    freq_penalty = tl.load(frequency_penalty_ptr + req_state_idx)
    pres_penalty = tl.load(presence_penalty_ptr + req_state_idx)

    use_rep_penalty = rep_penalty != 1.0
    use_freq_penalty = freq_penalty != 0.0
    use_pres_penalty = pres_penalty != 0.0
    use_penalty = (use_rep_penalty or use_freq_penalty) or use_pres_penalty
    if not use_penalty:
        # Early return to avoid loading logits.
        return

    block_idx = tl.program_id(1)
    block = block_idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = block < vocab_size
    logits = tl.load(logits_ptr + token_idx * logits_stride + block, mask=mask)
    logits = logits.to(tl.float32)

    base_output_counts = tl.load(
        output_bin_counts_ptr + req_state_idx * output_bin_counts_stride + block,
        mask=mask,
        other=0,
    )

    # Compute cumulative draft_counts from previous positions in this request
    pos = tl.load(expanded_local_pos_ptr + token_idx)
    start_idx = token_idx - pos
    draft_counts = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    for prev_pos in tl.static_range(MAX_SPEC_LEN):
        if prev_pos < pos:
            prev_token = tl.load(token_ids_ptr + start_idx + prev_pos + 1)
            token_match = block == prev_token
            draft_counts = draft_counts + token_match.to(tl.int32)

    # Total counts = base output counts + cumulative draft counts
    output_bin_counts = base_output_counts + draft_counts
    output_bin_mask = output_bin_counts > 0

    # Apply repetition penalties.
    if use_rep_penalty:
        packed_block = block_idx * BLOCK_SIZE // 32 + tl.arange(0, BLOCK_SIZE // 32)
        packed_mask = tl.load(
            prompt_bin_mask_ptr + req_state_idx * prompt_bin_mask_stride + packed_block,
            mask=packed_block < tl.cdiv(vocab_size, 32),
            other=0,
        )
        prompt_bin_mask = (packed_mask[:, None] >> (tl.arange(0, 32)[None, :])) & 1
        prompt_bin_mask = prompt_bin_mask.to(tl.int1)
        prompt_bin_mask = prompt_bin_mask.reshape(BLOCK_SIZE)

        # If token appears in prompt or output, apply, otherwise use 1.0 for no-op.
        scale = tl.where(prompt_bin_mask | output_bin_mask, rep_penalty, 1.0)
        # If logits are positive, divide by penalty, otherwise multiply by penalty.
        logits *= tl.where(logits > 0, 1.0 / scale, scale)

    # Apply frequency penalties.
    logits -= freq_penalty * output_bin_counts
    # Apply presence penalties.
    logits -= pres_penalty * output_bin_mask
    # Store back to logits.
    tl.store(logits_ptr + token_idx * logits_stride + block, logits, mask=mask)


import vllm.v1.worker.gpu.block_table

vllm.v1.worker.gpu.block_table._gather_block_tables_kernel = _gather_block_tables_kernel
vllm.v1.worker.gpu.block_table._compute_slot_mappings_kernel = (
    _compute_slot_mappings_kernel
)

import vllm.v1.worker.gpu.sample.penalties

vllm.v1.worker.gpu.sample.penalties._penalties_kernel = _penalties_kernel
