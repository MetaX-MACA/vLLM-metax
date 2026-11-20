# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch

from vllm_metax.v1.attention.ops.bf16_mqa_logits import bf16_mqa_logits

pytestmark = pytest.mark.requires_maca


@pytest.mark.parametrize("clean_logits", [False, True])
@pytest.mark.parametrize("length_format", ["flat", "request", "per-query"])
def test_bf16_indexer_logits_match_masked_reference(
    device, clean_logits, length_format
):
    torch.manual_seed(0)
    num_rows, heads, dim, max_length = 6, 40, 64, 80
    q = torch.randn(num_rows, heads, dim, device=device, dtype=torch.bfloat16)
    weights = torch.randn(num_rows, heads, device=device)
    kwargs = {}
    if length_format == "flat":
        k = torch.randn(max_length, dim, device=device, dtype=q.dtype)
        starts = torch.tensor([0, 15, 3, 60, 71, 0], device=device, dtype=torch.int32)
        ends = torch.tensor([30, 65, 9, 75, 71, 80], device=device, dtype=torch.int32)
        keys_by_row = k.expand(num_rows, -1, -1)
        limits = ends
    else:
        k = torch.randn(10, 16, 1, dim, device=device, dtype=q.dtype)
        table = torch.tensor(
            [[6, 2, 7, 0, 4], [9, 1, 5, 3, 8]], device=device, dtype=torch.int32
        )
        kwargs = {"block_table": table, "max_model_len": max_length}
        q = q.reshape(2, 3, heads, dim)
        starts = torch.zeros(num_rows, device=device, dtype=torch.int32)
        if length_format == "request":
            ends = torch.tensor([73, 15], device=device, dtype=torch.int32)
            limits = (ends[:, None] - torch.arange(2, -1, -1, device=device)).flatten()
        else:
            ends = torch.tensor(
                [[0, 67, 73], [14, 0, 15]], device=device, dtype=torch.int32
            )
            limits = ends.flatten()
        keys_by_row = (
            k[table.long()].reshape(2, max_length, dim).repeat_interleave(3, dim=0)
        )

    actual = bf16_mqa_logits(
        q, k, weights, starts, ends, clean_logits=clean_logits, **kwargs
    )
    scores = torch.einsum(
        "rhd,rnd->rhn", q.reshape(num_rows, heads, dim).float(), keys_by_row.float()
    )
    expected = (scores.relu() * weights[..., None]).sum(dim=1)
    cols = torch.arange(max_length, device=device)
    valid = (cols >= starts[:, None]) & (cols < limits[:, None])
    torch.testing.assert_close(actual[valid], expected[valid], atol=3e-4, rtol=3e-5)
    if clean_logits:
        assert actual[~valid].isneginf().all()
    else:
        assert (actual[~valid] == 0).all()
