# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# -----------------------------------------------------------------------------
# Purpose: run speculative-decode rejection resampling with a 512-token block
#       on MACA.
# Added version: v0.30.0.
#
# Note: Cap the resample block size of the speculative-decode rejection
#       sampler at 512 on MACA. With the upstream RESAMPLE_BLOCK_SIZE=1024,
#       MTP rejection sampling intermittently hits device-side memory
#       violations on MACA; 512 is stable.
#
#       The replacement is installed on the provider module
#       (rejection_sampler_utils). The shared patch utility synchronizes the
#       caller's `from ...rejection_sampler_utils import rejection_sample`
#       alias, so the 512 implementation is used regardless of import order.
#
#       Preserve the MC3-15766 four-clamp insert workaround around expanded
#       logits, request state, resample blocks and sampled output slots.
#
# Affected versions: Historical MACA reproductions on vLLM 0.29.0/0.30.0;
#     source and launch contracts reviewed on v0.32.0 (5df63a7203), 2026-11-03.
#     This review does not establish native GPU numerical correctness.
#
# Remove at: MACA runs `_resample_kernel` reliably with
#     RESAMPLE_BLOCK_SIZE=1024, or upstream makes the resample block size
#     platform-tunable; retire insert clamps after equivalent upstream bounds
#     or a verified fix for MC3-15766.
# -----------------------------------------------------------------------------

import torch
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils import (
    _compute_cumulative_log_p_kernel,
    _compute_local_logits_stats_kernel,
    _compute_local_residual_mass_kernel,
    _rejection_kernel,
    _resample_kernel,
)
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
    idx_in_range = tl.minimum(
        tl.maximum(resample_token_idx, 0), tl.maximum(num_logits - 1, 0)
    )
    req_state_idx = tl.load(expanded_idx_mapping_ptr + idx_in_range)
    # Increment the number of sampled tokens.
    tl.store(num_sampled_ptr + req_idx, num_sampled + 1)
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
    slot_in_range = tl.minimum(
        tl.maximum(num_sampled, 0), tl.maximum(sampled_stride - 1, 0)
    )
    tl.store(
        sampled_ptr + req_idx * sampled_stride + slot_in_range,
        resampled,
    )
    # \-------------------- MetaX Modification --------------------/


@patch("vllm.v1.worker.gpu.spec_decode.rejection_sampler_utils", "rejection_sample")
def rejection_sample(
    # [num_logits, V]
    target_logits: torch.Tensor,
    # [max_num_reqs, num_speculative_steps, V]
    draft_logits: torch.Tensor | None,
    # [num_logits]
    draft_sampled: torch.Tensor,
    # [num_reqs + 1]
    cu_num_logits: torch.Tensor,
    # [num_logits]
    pos: torch.Tensor,
    # [num_reqs]
    idx_mapping: torch.Tensor,
    # [num_logits]
    expanded_idx_mapping: torch.Tensor,
    # [num_logits]
    expanded_local_pos: torch.Tensor,
    # [max_num_reqs]
    temperature: torch.Tensor,
    # [max_num_reqs]
    seed: torch.Tensor,
    num_speculative_steps: int,
    # [num_speculative_steps]
    synthetic_conditional_rates: torch.Tensor | None = None,
    use_fp64: bool = False,
    use_block_verification: bool = False,
    contexts: torch.Tensor | None = None,
    watermarking: torch.Tensor | None = None,
    watermarking_skip_mask: torch.Tensor | None = None,
    watermark_key: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert target_logits.ndim == 2 and target_logits.stride(-1) == 1
    assert draft_logits is None or (
        draft_logits.ndim == 3 and draft_logits.stride(-1) == 1
    )
    num_reqs = cu_num_logits.shape[0] - 1
    num_logits, vocab_size = target_logits.shape

    watermark = contexts is not None
    assert watermark == (watermarking is not None) == (watermark_key is not None), (
        "contexts, watermarking and watermark_key must be set together."
    )
    assert watermarking_skip_mask is None or watermark, (
        "watermarking_skip_mask requires watermarking."
    )
    contexts_stride = 0
    context_width = 1
    watermarking_bytes: torch.Tensor | None = None
    watermarking_skip_mask_bytes: torch.Tensor | None = None
    watermark_key_0 = 0
    watermark_key_1 = 0
    if contexts is not None:
        assert watermarking is not None and watermark_key is not None
        assert contexts.ndim == 2 and contexts.shape[0] == num_logits
        # Context words are hashed as uint32, so int32 (the request-state token
        # dtype) and int64 both land on the same PRF stream, -1 included.
        assert contexts.dtype in (torch.int32, torch.int64)
        if contexts.stride(-1) != 1:
            contexts = contexts.contiguous()
        assert watermarking.ndim == 1 and watermarking.dtype == torch.bool
        contexts_stride = contexts.stride(0)
        context_width = contexts.shape[-1]
        watermarking_bytes = watermarking.view(torch.uint8)
        if watermarking_skip_mask is not None:
            assert watermarking_skip_mask.shape == (num_logits,)
            assert watermarking_skip_mask.dtype == torch.bool
            if not watermarking_skip_mask.is_contiguous():
                watermarking_skip_mask = watermarking_skip_mask.contiguous()
            watermarking_skip_mask_bytes = watermarking_skip_mask.view(torch.uint8)
        watermark_key_0 = watermark_key & 0xFFFFFFFF
        watermark_key_1 = watermark_key >> 32
    draft_logits_stride_0 = 0
    draft_logits_stride_1 = 0
    if has_draft_logits := draft_logits is not None:
        draft_logits_stride_0 = draft_logits.stride(0)
        draft_logits_stride_1 = draft_logits.stride(1)
        # In some cases (e.g. MiMo v2.5 Pro + DFlash) the target model's
        # vocab size is larger than the draft's due to padding.
        vocab_size = min(vocab_size, draft_logits.size(-1))

    # Compute the per-vocab-block logits stats, such as target argmax
    # (for greedy requests), and target max + softmax exponential
    # (for non-greedy requests).
    VOCAB_BLOCK_SIZE = 8192
    vocab_num_blocks = triton.cdiv(vocab_size, VOCAB_BLOCK_SIZE)
    padded_vocab_num_blocks = triton.next_power_of_2(vocab_num_blocks)
    target_local_argmax = target_logits.new_empty(
        num_logits, vocab_num_blocks, dtype=torch.int64
    )
    target_local_max = target_logits.new_empty(
        num_logits, vocab_num_blocks, dtype=torch.float32
    )
    target_local_sumexp = target_logits.new_empty(
        num_logits, vocab_num_blocks, dtype=torch.float32
    )
    draft_local_max = target_logits.new_empty(
        num_logits, vocab_num_blocks, dtype=torch.float32
    )
    draft_local_sumexp = target_logits.new_empty(
        num_logits, vocab_num_blocks, dtype=torch.float32
    )
    _compute_local_logits_stats_kernel[(num_logits, vocab_num_blocks)](
        target_local_argmax,
        target_local_argmax.stride(0),
        target_local_max,
        target_local_max.stride(0),
        target_local_sumexp,
        target_local_sumexp.stride(0),
        draft_local_max,
        draft_local_max.stride(0),
        draft_local_sumexp,
        draft_local_sumexp.stride(0),
        target_logits,
        target_logits.stride(0),
        draft_logits,
        draft_logits_stride_0,
        draft_logits_stride_1,
        expanded_idx_mapping,
        expanded_local_pos,
        temperature,
        vocab_size,
        num_speculative_steps,
        BLOCK_SIZE=VOCAB_BLOCK_SIZE,
        HAS_DRAFT_LOGITS=has_draft_logits,
    )

    # Precompute the running joint ratio and residual mass for block
    # verification.
    if use_block_verification:
        assert synthetic_conditional_rates is None, (
            "Block verification is incompatible with synthetic acceptance rates."
        )

        # Compute the log of the running joint ratio, p_i.
        # cumulative_log_p[start + i] = log(p_{i+1}), the cumulative ratio after
        # the (i+1)-th draft token.
        cumulative_log_p = target_logits.new_empty(num_logits, dtype=torch.float32)
        _compute_cumulative_log_p_kernel[(num_reqs,)](
            cumulative_log_p,
            target_logits,
            target_logits.stride(0),
            target_local_max,
            target_local_max.stride(0),
            target_local_sumexp,
            target_local_sumexp.stride(0),
            draft_sampled,
            draft_logits,
            draft_logits_stride_0,
            draft_logits_stride_1,
            draft_local_max,
            draft_local_max.stride(0),
            draft_local_sumexp,
            draft_local_sumexp.stride(0),
            cu_num_logits,
            idx_mapping,
            temperature,
            vocab_num_blocks,
            PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
            HAS_DRAFT_LOGITS=has_draft_logits,
            num_warps=1,
        )

        # Compute the per-vocab-block partials of the residual mass, later reduced
        # to the total by _compute_global_residual_mass. Only launched for full
        # draft logits distributions. One-hot drafts used a closed-form residual
        # mass instead.
        if has_draft_logits:
            local_residual_mass = target_logits.new_empty(
                num_logits, vocab_num_blocks, dtype=torch.float32
            )
            _compute_local_residual_mass_kernel[(num_logits, vocab_num_blocks)](
                local_residual_mass,
                local_residual_mass.stride(0),
                cumulative_log_p,
                target_logits,
                target_logits.stride(0),
                target_local_max,
                target_local_max.stride(0),
                target_local_sumexp,
                target_local_sumexp.stride(0),
                draft_logits,
                draft_logits_stride_0,
                draft_logits_stride_1,
                draft_local_max,
                draft_local_max.stride(0),
                draft_local_sumexp,
                draft_local_sumexp.stride(0),
                draft_sampled,
                expanded_idx_mapping,
                expanded_local_pos,
                temperature,
                vocab_size,
                num_speculative_steps,
                vocab_num_blocks,
                BLOCK_SIZE=VOCAB_BLOCK_SIZE,
                PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
            )
        else:
            local_residual_mass = None
    else:
        cumulative_log_p = None
        local_residual_mass = None

    # Sample up until the first rejected/bonus token, and store
    # the step.
    sampled = draft_sampled.new_empty(
        num_reqs, num_speculative_steps + 1, dtype=torch.int64
    )
    num_sampled = sampled.new_empty(num_reqs, dtype=torch.int32)
    target_rejected_logsumexp = target_logits.new_empty(num_reqs, dtype=torch.float32)
    draft_rejected_logsumexp = target_logits.new_empty(num_reqs, dtype=torch.float32)
    _rejection_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        target_rejected_logsumexp,
        draft_rejected_logsumexp,
        target_logits,
        target_logits.stride(0),
        target_local_argmax,
        target_local_argmax.stride(0),
        target_local_max,
        target_local_max.stride(0),
        target_local_sumexp,
        target_local_sumexp.stride(0),
        draft_sampled,
        draft_logits,
        draft_logits_stride_0,
        draft_logits_stride_1,
        draft_local_max,
        draft_local_max.stride(0),
        draft_local_sumexp,
        draft_local_sumexp.stride(0),
        cu_num_logits,
        idx_mapping,
        temperature,
        seed,
        pos,
        synthetic_conditional_rates,
        cumulative_log_p,
        local_residual_mass,
        local_residual_mass.stride(0) if local_residual_mass is not None else 0,
        vocab_num_blocks,
        PADDED_VOCAB_NUM_BLOCKS=padded_vocab_num_blocks,
        HAS_DRAFT_LOGITS=has_draft_logits,
        SYNTHETIC_MODE=synthetic_conditional_rates is not None,
        USE_BLOCK_VERIFICATION=use_block_verification,
        num_warps=1,
    )

    # Resample the rejected/bonus tokens.
    # /-------------------- MetaX Modification --------------------\
    # Upstream uses 1024, which intermittently triggers device-side
    # memory violations on MACA; 512 is stable.
    RESAMPLE_BLOCK_SIZE = 512
    # \-------------------- MetaX Modification --------------------/
    resample_num_blocks = triton.cdiv(vocab_size, RESAMPLE_BLOCK_SIZE)
    padded_resample_num_blocks = triton.next_power_of_2(resample_num_blocks)
    resampled_local_argmax = target_logits.new_empty(
        num_reqs, resample_num_blocks, dtype=torch.int64
    )
    resampled_local_max = target_logits.new_empty(
        num_reqs,
        resample_num_blocks,
        dtype=torch.float64 if use_fp64 else torch.float32,
    )
    _resample_kernel[(num_reqs, resample_num_blocks)](
        resampled_local_argmax,
        resampled_local_argmax.stride(0),
        resampled_local_max,
        resampled_local_max.stride(0),
        target_logits,
        target_logits.stride(0),
        target_rejected_logsumexp,
        draft_logits,
        draft_logits_stride_0,
        draft_logits_stride_1,
        draft_rejected_logsumexp,
        num_sampled,
        cu_num_logits,
        expanded_idx_mapping,
        draft_sampled,
        temperature,
        seed,
        pos,
        cumulative_log_p,
        contexts,
        contexts_stride,
        watermarking_bytes,
        watermarking_skip_mask_bytes,
        watermark_key_0,
        watermark_key_1,
        vocab_size,
        BLOCK_SIZE=RESAMPLE_BLOCK_SIZE,
        HAS_DRAFT_LOGITS=has_draft_logits,
        USE_FP64=use_fp64,
        USE_BLOCK_VERIFICATION=use_block_verification,
        CONTEXT_WIDTH=context_width,
        WATERMARK=watermark,
        DEDUPLICATE_CONTEXTS=watermarking_skip_mask is not None,
    )

    # Insert the resampled tokens into the output sampled.
    # Use the MC3-15766 bounds kernel installed above.
    _insert_resampled_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        resampled_local_argmax,
        resampled_local_argmax.stride(0),
        resampled_local_max,
        resampled_local_max.stride(0),
        resample_num_blocks,
        cu_num_logits,
        expanded_idx_mapping,
        temperature,
        PADDED_RESAMPLE_NUM_BLOCKS=padded_resample_num_blocks,
    )
    return sampled, num_sampled
