# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch

from vllm_metax.platform import MacaPlatform


@pytest.mark.requires_maca
def test_ngram_compute_n_gram_ids(device):
    MacaPlatform.import_kernels()

    def i32(data):
        return torch.tensor(data, dtype=torch.int32, device=device)

    weights = [[[3, 5, 7], [11, 13, 17]], [[19, 23, 27], [29, 31, 37]]]
    mods = [[13, 17], [23, 29]]
    embedding_offsets = [0, 13, 30, 53, 82]
    request_starts = [0, 4, 7, 7]
    rows = [2, 0, 1]
    columns = [0, 3, 0]
    tokens = [[6, 4, 7, 5, 3, 2], [0, 0, 0, 0, 0, 0], [3, -1, 5, 8, -1, 9]]
    result = torch.full((request_starts[-1], 4), -1, dtype=torch.int32, device=device)
    torch.ops._C.ngram_compute_n_gram_ids(
        3,
        2,
        i32(weights),
        i32(mods),
        i32(embedding_offsets),
        i32(request_starts),
        i32(tokens),
        torch.tensor(rows, dtype=torch.int64, device=device),
        i32(columns),
        result,
    )

    expected = []
    for request, row in enumerate(rows):
        for column in range(
            columns[request],
            columns[request] + request_starts[request + 1] - request_starts[request],
        ):
            ids = []
            for n in range(2):
                for k in range(2):
                    value = 0
                    for lag, weight in enumerate(weights[n][k][: n + 2]):
                        if column < lag or tokens[row][column - lag] < 0:
                            break
                        value += weight * tokens[row][column - lag]
                    ids.append(value % mods[n][k] + embedding_offsets[n * 2 + k])
            expected.append(ids)

    torch.testing.assert_close(result, i32(expected), atol=0, rtol=0)
