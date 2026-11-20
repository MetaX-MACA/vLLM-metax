# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import math

import pytest
import torch

from vllm_metax.v1.attention.ops.flashmla import flash_mla_sparse_fwd

pytestmark = pytest.mark.requires_maca


# Adapted from vLLM v0.33.0 tests/kernels/attention/test_flashmla_sparse.py.
@pytest.mark.parametrize("heads", [64, 128])
def test_flashmla_sparse_prefill_respects_active_indices(device, heads):
    torch.manual_seed(0)
    num_queries, dim, value_dim = 3, 576, 512
    query = torch.randn(num_queries, heads, dim, device=device, dtype=torch.bfloat16)
    kv = torch.randn(144, 1, dim, device=device, dtype=query.dtype)
    indices = torch.randperm(len(kv), device=device, dtype=torch.int32)[:128]
    indices = indices.expand(num_queries, 1, -1).contiguous()
    lengths = [1, 17, 128]
    scale = dim**-0.5
    output, max_logits, lse = flash_mla_sparse_fwd(
        query,
        kv,
        indices,
        scale,
        value_dim,
        topk_length=torch.tensor(lengths, device=device, dtype=torch.int32),
    )
    expected_output, expected_max, expected_lse = [], [], []
    for row, length in enumerate(lengths):
        keys = kv[indices[row, 0, :length].long(), 0].float()
        logits = (query[row].float() @ keys.T) * scale
        expected_output.append(logits.softmax(-1) @ keys[:, :value_dim])
        expected_max.append(logits.amax(-1))
        expected_lse.append(logits.logsumexp(-1))
    torch.testing.assert_close(
        output, torch.stack(expected_output).to(query.dtype), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        max_logits, torch.stack(expected_max) * math.log2(math.e), atol=3e-3, rtol=3e-3
    )
    torch.testing.assert_close(lse, torch.stack(expected_lse), atol=3e-3, rtol=3e-3)
