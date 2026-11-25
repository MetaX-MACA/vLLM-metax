# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch

from vllm.v1.worker.gpu.spec_decode import rejection_sampler, rejection_sampler_utils
from vllm.v1.worker.gpu.spec_decode.draft_logits import DraftLogitsCache

from vllm_metax.patch.bugfix.triton_support.rejection_sample import rejection_sample


# Adapted from vLLM v0.33.0 tests/v1/spec_decode/test_rejection_sampler_utils.py.
@pytest.mark.requires_maca
@pytest.mark.parametrize("temperature", [0.0, 1.0], ids=["greedy", "point-mass"])
@pytest.mark.parametrize("block_verification", [False, True])
@pytest.mark.parametrize("skip_masked_vocab_blocks", [False, True])
def test_patched_rejection_sampling_handles_short_and_rejected_requests(
    device, temperature, block_verification, skip_masked_vocab_blocks
):
    assert rejection_sampler_utils.rejection_sample is rejection_sample
    assert rejection_sampler.rejection_sample is rejection_sample
    lengths = [4, 3, 1]
    target_tokens = [19, 600, 1025, 1500, 1026, 70, 17, 730]
    num_logits = sum(lengths)
    vocab_size = 1537
    logits = torch.full((num_logits, vocab_size), -torch.inf, device=device)
    logits[torch.arange(num_logits, device=device), target_tokens] = 0
    state = torch.tensor([5, 1, 2], device=device, dtype=torch.int32)
    draft_logits_cache = None
    if block_verification:
        draft_logits_cache = DraftLogitsCache(
            6, 3, vocab_size, dtype=logits.dtype, fill=-torch.inf, device=device
        )
        draft_logits_cache.row_mapping = torch.tensor(
            [-1, 0, 2, -1, -1, 1], device=device, dtype=torch.int32
        )
        draft_steps = torch.arange(3, device=device)
        draft_logits_cache.values[1, draft_steps, [19, 600, 1025]] = 0
        draft_logits_cache.values[0, draft_steps, [7, 70, 17]] = 0
    sampled, counts = rejection_sampler.rejection_sample(
        target_logits=logits,
        draft_logits_cache=draft_logits_cache,
        draft_sampled=torch.tensor([0, 19, 600, 1025, 0, 7, 70, 0], device=device),
        cu_num_logits=torch.tensor([0, 4, 7, 8], device=device, dtype=torch.int32),
        pos=torch.arange(num_logits, device=device, dtype=torch.int32),
        idx_mapping=state,
        expanded_idx_mapping=state.repeat_interleave(
            torch.tensor(lengths, device=device)
        ),
        expanded_local_pos=torch.tensor(
            [0, 1, 2, 3, 0, 1, 2, 0], device=device, dtype=torch.int32
        ),
        temperature=torch.full((6,), temperature, device=device),
        seed=torch.arange(6, device=device, dtype=torch.int64),
        num_speculative_steps=3,
        use_block_verification=block_verification,
        skip_masked_vocab_blocks=skip_masked_vocab_blocks,
    )
    assert counts.tolist() == [4, 1, 1]
    assert sampled[0, :4].tolist() == target_tokens[:4]
    assert sampled[1, 0].item() == target_tokens[4]
    assert sampled[2, 0].item() == target_tokens[-1]
