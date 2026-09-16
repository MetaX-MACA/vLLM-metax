# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: `_compute_global_topk_indices_and_lens_kernel` gathers block numbers
#       from the block table using `req_idx` (per-token request row) and
#       `local_idx // block_size` (block-within-row column) without any bounds
#       check. On decode steps whose request geometry changed (CUDA-graph
#       padding/warmup or mixed decode+prefill batches), `token_to_req_indices`
#       can carry a request row that no longer has a corresponding row in the
#       sliced `block_table[:num_decodes]`, and stale local top-k indices can
#       point past the row's block count. The unmasked gather then faults at
#       the hardware level (NVIDIA: cudaErrorIllegalAddress in
#       `_compute_global_topk_indices_and_lens_kernel`; MACA: Xnack/ATU
#       address-translation trap on the same kernel), reproduced with
#       DeepSeek-V4-Flash W8A8 @ batch 32.
#
#       Upstream tracks the same defect in vllm-project/vllm#55636 and the two
#       candidate fixes #55692 / #55744. This backport combines both
#       consumer-hardening layers:
#       1. Reject `req_idx` outside [0, block_table.shape[0]) and block
#          indices outside [0, block_table.shape[1]) *before* the gather.
#       2. Clamp the pointer operands instead of only masking the load:
#          MetaX's mcTriton runtime can raise an ATU/Xnack fault for
#          masked-off lanes too when the formed address is out of bounds.
#       3. Pack surviving entries to the front of each row. Downstream
#          `flash_fwd_splitkv_sparse_mla_kernel` consumes the first
#          `topk_lens` entries as a prefix; an interior -1 inside that prefix
#          dereferences slot -1 and faults with a Memory Violation (0x4).
#
# Affected versions: v0.27.0. The kernel body is unchanged since
#       vllm-project/vllm releases/v0.26.0, but the wrapper gained the
#       `output_buffers` argument in v0.27.0 (v0.26.0 has the 5-argument
#       form); this patch keeps the v0.27.0 signature, which v0.28.0 shares.
#
# Remove at: vLLM dependency bumped past the merged upstream fix for #55636
#            (neither #55692 nor #55744 was merged as of 2026-09-08).
# -----------------------------------------------------------------------------

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

from vllm_metax.patch.utils import patch

logger = init_logger("vllm.vllm_metax.patch.bugfix.deepseek_v4.topk_global_indices_oob")
_guard_logged = False


@patch(
    "vllm.models.deepseek_v4.common.ops.cache_utils",
    "_compute_global_topk_indices_and_lens_kernel",
)
@triton.jit(do_not_specialize=["block_table_rows", "block_table_cols"])
def _compute_global_topk_indices_and_lens_kernel(
    global_topk_indices_ptr,
    global_topk_indices_stride: tl.constexpr,
    topk_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride: tl.constexpr,
    topk: tl.constexpr,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_rows,
    block_table_cols,
    block_table_stride: tl.constexpr,
    block_size: tl.constexpr,
    is_valid_token_ptr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    is_valid_token = tl.load(is_valid_token_ptr + token_idx)
    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    # /-------------------- MetaX Modification --------------------\
    req_idx_valid = (req_idx >= 0) & (req_idx < block_table_rows)
    # Clamp before forming the gather address: masked-off lanes can still be
    # address-translated by the MACA runtime, so an out-of-bounds pointer
    # must never be constructed.
    safe_req_idx = tl.where(req_idx_valid, req_idx, 0)
    # \-------------------- MetaX Modification --------------------/

    count = tl.zeros((), dtype=tl.int32)
    for i in range(0, topk, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        mask = offset < topk

        local_idx = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + offset,
            mask=mask,
            other=-1,
        )
        block_indices = local_idx // block_size
        # /-------------------- MetaX Modification --------------------\
        # Mask the block-table gather before it is issued: a stale request
        # row (CUDA-graph padding / geometry change) or a block index past
        # the row's allocated columns must never be dereferenced.
        is_valid = (
            (local_idx >= 0)
            & req_idx_valid
            & is_valid_token
            & (block_indices < block_table_cols)
        )
        # \-------------------- MetaX Modification --------------------/
        # /-------------------- MetaX Modification --------------------\
        safe_block_indices = tl.where(is_valid, block_indices, 0)
        # \-------------------- MetaX Modification --------------------/
        block_numbers = tl.load(
            block_table_ptr + safe_req_idx * block_table_stride + safe_block_indices,
            mask=mask & is_valid,
            other=0,
        )
        block_offsets = local_idx % block_size

        slot_ids = block_numbers * block_size + block_offsets
        slot_ids = tl.where(is_valid, slot_ids, -1)
        # /-------------------- MetaX Modification --------------------\
        # Downstream reads the first topk_lens entries as a packed prefix.
        # Prefill the row with -1 and place survivors at the front so an
        # interior invalid entry can never be dereferenced as slot -1.
        tl.store(
            global_topk_indices_ptr + token_idx * global_topk_indices_stride + offset,
            -1,
            mask=mask,
        )
        valid_i32 = is_valid.to(tl.int32)
        packed_offset = count + tl.cumsum(valid_i32, axis=0) - valid_i32
        tl.store(
            global_topk_indices_ptr
            + token_idx * global_topk_indices_stride
            + packed_offset,
            slot_ids,
            mask=mask & is_valid,
        )
        count += tl.sum(valid_i32, axis=0)
        # \-------------------- MetaX Modification --------------------/

    # Zero out length for padding tokens.
    tl.store(topk_lens_ptr + token_idx, tl.where(is_valid_token, count, 0))


@patch(
    "vllm.models.deepseek_v4.common.ops.cache_utils",
    "compute_global_topk_indices_and_lens",
)
@patch(
    "vllm.models.deepseek_v4.common.ops",
    "compute_global_topk_indices_and_lens",
)
def compute_global_topk_indices_and_lens(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
    output_buffers: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map local topk indices to global KV cache slots and count valid entries.

    Fuses three operations into a single kernel:
    1. Block-table lookup (local index → global slot id)
    2. Valid-entry counting (topk_lens per token)
    3. Masking padding tokens to length 0
    """
    # /-------------------- MetaX Modification --------------------\
    global _guard_logged
    if not _guard_logged:
        _guard_logged = True
        logger.info(
            "vllm_metax topk global-indices OOB guard active (rows=%s, cols=%s)",
            block_table.shape[0],
            block_table.shape[1],
        )
    # \-------------------- MetaX Modification --------------------/
    num_tokens = topk_indices.shape[0]
    if output_buffers is None:
        global_topk_indices = torch.empty_like(topk_indices)
        topk_lens = torch.empty(
            num_tokens, dtype=torch.int32, device=topk_indices.device
        )
    else:
        global_topk_indices, topk_lens = output_buffers
        assert global_topk_indices.shape == topk_indices.shape
        assert topk_lens.shape == (num_tokens,)
    _compute_global_topk_indices_and_lens_kernel[(num_tokens,)](
        global_topk_indices,
        global_topk_indices.stride(0),
        topk_lens,
        topk_indices,
        topk_indices.stride(0),
        topk_indices.shape[-1],
        token_to_req_indices,
        block_table,
        # /-------------------- MetaX Modification --------------------\
        block_table.shape[0],
        block_table.shape[1],
        # \-------------------- MetaX Modification --------------------/
        block_table.stride(0),
        block_size,
        is_valid_token,
        TRITON_BLOCK_SIZE=1024,
    )
    return global_topk_indices, topk_lens
