# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

import pytest
import torch

from vllm_metax.v1.attention.ops.merge_attn_states import merge_attn_states

pytestmark = pytest.mark.requires_maca


# Adapted from vLLM v0.33.0 tests/kernels/attention/test_merge_attn_states.py.
@pytest.mark.parametrize("head_size", [64, 63], ids=["native", "triton-fallback"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("context_tokens", [None, 0, 3])
@pytest.mark.parametrize("return_lse", [True, False])
@torch.inference_mode()
def test_merge_attention_with_empty_partitions(
    device, head_size, dtype, context_tokens, return_lse
):
    torch.manual_seed(0)
    prefix = torch.randn(7, 2, head_size, device=device, dtype=dtype)
    suffix = torch.randn_like(prefix)
    prefix_lse = torch.randn(2, 7, device=device)
    suffix_lse = torch.randn_like(prefix_lse)
    # Empty attention partitions have zero output and an infinite LSE sentinel.
    prefix[0].zero_()
    suffix[0].zero_()
    prefix_lse[:, :2] = torch.inf
    suffix_lse[:, 0] = -torch.inf
    suffix_lse[:, 2] = torch.inf
    actual = torch.empty_like(prefix)
    actual_lse = torch.empty_like(prefix_lse) if return_lse else None
    merge_attn_states(
        actual,
        prefix,
        prefix_lse,
        suffix,
        suffix_lse,
        actual_lse,
        context_tokens,
    )

    end = len(prefix) if context_tokens is None else context_tokens
    logits = torch.stack([prefix_lse, suffix_lse])
    logits.masked_fill_(logits.isposinf(), -torch.inf)
    weights = logits.softmax(dim=0).nan_to_num().transpose(1, 2).unsqueeze(-1)
    expected = (weights * torch.stack([prefix, suffix]).float()).sum(0).to(dtype)
    expected[end:] = suffix[end:]
    torch.testing.assert_close(actual, expected, atol=4e-3, rtol=4e-3)
    if return_lse:
        expected_lse = logits.logsumexp(dim=0)
        expected_lse[:, end:] = suffix_lse[:, end:]
        torch.testing.assert_close(actual_lse, expected_lse, atol=1e-6, rtol=1e-6)
