# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# Note: MC3-15766: preserve the four-clamp insert workaround validated on MetaX
#     and skip insertion when the request has no remaining resample token.
# Affected versions: vLLM 0.29.0 on MetaX.
# Remove at: upstream provides equivalent insert bounds, or the MetaX backend
#     illegal-access issue is resolved and the workaround is no longer needed.

from vllm.triton_utils import tl, triton

from vllm_metax.patch.utils import patch


@patch(
    "vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils",
    "_insert_resampled_kernel",
)
@triton.heuristics(
    {
        "num_logits": lambda args: args["expanded_idx_mapping_ptr"].numel(),
        "max_num_reqs": lambda args: args["temp_ptr"].shape[0],
    }
)
@triton.jit
def _insert_resampled_kernel(
    # [num_reqs, num_speculative_steps + 1]
    sampled_ptr,
    sampled_stride,
    # [num_reqs]
    num_sampled_ptr,
    # [num_reqs, num_blocks]
    resampled_local_argmax_ptr,
    resampled_local_argmax_stride,
    # [num_reqs, num_blocks]
    resampled_local_max_ptr,
    resampled_local_max_stride,
    resample_num_blocks,
    # [num_reqs + 1]
    cu_num_logits_ptr,
    # [num_reqs]
    expanded_idx_mapping_ptr,
    # [max_num_reqs]
    temp_ptr,
    # /-------------------- MetaX Modification --------------------\
    num_logits,
    max_num_reqs,
    # \-------------------- MetaX Modification --------------------/
    PADDED_RESAMPLE_NUM_BLOCKS: tl.constexpr,
):
    req_idx = tl.program_id(0)
    num_sampled = tl.load(num_sampled_ptr + req_idx)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)

    resample_token_idx = start_idx + num_sampled
    # /-------------------- MetaX Modification --------------------\
    if resample_token_idx >= end_idx:
        return

    idx_in_range = tl.minimum(
        tl.maximum(resample_token_idx, 0), tl.maximum(num_logits - 1, 0)
    )
    req_state_idx = tl.load(expanded_idx_mapping_ptr + idx_in_range)
    # \-------------------- MetaX Modification --------------------/

    # Increment the number of sampled tokens.
    tl.store(num_sampled_ptr + req_idx, num_sampled + 1)

    # /-------------------- MetaX Modification --------------------\
    state_in_range = tl.minimum(
        tl.maximum(req_state_idx, 0), tl.maximum(max_num_reqs - 1, 0)
    )
    temp = tl.load(temp_ptr + state_in_range).to(tl.float32)
    # \-------------------- MetaX Modification --------------------/
    is_bonus = resample_token_idx == end_idx - 1
    if temp == 0.0 and not is_bonus:
        # Greedy + non-bonus token. The target argmax is already
        # in the sampled tensor.
        return

    # Insert the resampled token.
    block = tl.arange(0, PADDED_RESAMPLE_NUM_BLOCKS)
    mask = block < resample_num_blocks
    resampled_local_max = tl.load(
        resampled_local_max_ptr + req_idx * resampled_local_max_stride + block,
        mask=mask,
        other=float("-inf"),
    )
    # NaN max values (from NaN target logits) make tl.argmax return an
    # out-of-range block index (into the padded region), causing an OOB read
    # of resampled_local_argmax. Map NaN to -inf so argmax stays in range.
    resampled_local_max = tl.where(
        resampled_local_max != resampled_local_max,
        float("-inf"),
        resampled_local_max,
    )
    resampled_max_block_idx = tl.argmax(resampled_local_max, axis=0)
    # /-------------------- MetaX Modification --------------------\
    block_in_range = tl.minimum(
        resampled_max_block_idx, tl.maximum(resample_num_blocks - 1, 0)
    )
    resampled = tl.load(
        resampled_local_argmax_ptr
        + req_idx * resampled_local_argmax_stride
        + block_in_range,
    )
    # \-------------------- MetaX Modification --------------------/
    # /-------------------- MetaX Modification --------------------\
    slot_in_range = tl.minimum(
        tl.maximum(num_sampled, 0), tl.maximum(sampled_stride - 1, 0)
    )
    tl.store(
        sampled_ptr + req_idx * sampled_stride + slot_in_range,
        resampled,
    )
    # \-------------------- MetaX Modification --------------------/
