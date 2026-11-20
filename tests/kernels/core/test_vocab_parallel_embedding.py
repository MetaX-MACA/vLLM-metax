# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch

from vllm_metax.platform import MacaPlatform

pytestmark = pytest.mark.requires_maca


@pytest.mark.parametrize("id_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("hidden", [15, 16, 18, 20])
def test_rank_sum_is_full_lookup(device, id_dtype, hidden):
    MacaPlatform.import_kernels()
    ids = torch.tensor([[0, 6, 3, 10], [7, 1, 11, 4]], dtype=id_dtype, device=device)
    table = torch.randn(12, hidden, dtype=torch.bfloat16, device=device)
    total = torch.zeros(ids.numel(), hidden, dtype=table.dtype, device=device)
    shards = [(0, 5, 2, 7, 9), (5, 7, 3, 9, 12)]
    for start, end, padding, added_start, added_end in shards:
        gap = torch.full((padding, hidden), 999, dtype=table.dtype, device=device)
        weight = torch.cat((table[start:end], gap, table[added_start:added_end]))
        result = torch.empty_like(total)
        torch.ops._C.vocab_parallel_embedding(
            result, ids, weight, start, end, padding, added_start, added_end
        )
        owned = ((ids >= start) & (ids < end)) | (
            (ids >= added_start) & (ids < added_end)
        )
        reference = torch.where(
            owned.flatten()[:, None], table[ids.long().flatten()], 0
        )
        torch.testing.assert_close(result, reference, atol=0, rtol=0)
        total += result

    torch.testing.assert_close(total, table[ids.long().flatten()], atol=0, rtol=0)
