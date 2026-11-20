# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch

from vllm_metax.model_executor.layers.indexer_topk import SparseIndexerTopk
from vllm_metax.platform import MacaPlatform

pytestmark = pytest.mark.requires_maca


def _make_logits(device, num_rows, vocab_size):
    generator = torch.Generator().manual_seed(42)
    return torch.stack(
        [torch.randperm(vocab_size, generator=generator) for _ in range(num_rows)]
    ).to(device=device, dtype=torch.float32)


def _assert_topk_matches_reference(logits, indices, row_starts, row_ends):
    top_k = indices.size(1)
    for row, (start, end) in enumerate(zip(row_starts, row_ends, strict=True)):
        k = min(top_k, end - start)
        expected = logits[row, start:end].topk(k).indices.to(indices.dtype)
        actual = indices[row, :k]
        torch.testing.assert_close(actual.sort().values, expected.sort().values)
        assert (indices[row, k:] == -1).all()


@pytest.mark.parametrize("top_k", [512, 2048])
@torch.inference_mode()
def test_top_k_per_row(device, top_k):
    MacaPlatform.import_kernels()
    row_starts = [8, 16, 32]
    row_ends = [8, 23, 4105]
    logits = _make_logits(device, len(row_starts), 8192)
    indices = torch.full((logits.size(0), top_k), -2, device=device, dtype=torch.int32)
    torch.ops._C.top_k_per_row_prefill(
        logits,
        torch.tensor(row_starts, device=device, dtype=torch.int32),
        torch.tensor(row_ends, device=device, dtype=torch.int32),
        indices,
        logits.size(0),
        logits.stride(0),
        logits.stride(1),
        top_k,
    )
    _assert_topk_matches_reference(logits, indices, row_starts, row_ends)


def _run_top_k_per_row_decode_test(
    device, lengths, next_n, per_row_lengths, top_k, vocab_size
):
    MacaPlatform.import_kernels()
    row_ends = [length - next_n + i + 1 for length in lengths for i in range(next_n)]
    logits = _make_logits(device, len(row_ends), vocab_size)
    indices = torch.full((logits.size(0), top_k), -2, device=device, dtype=torch.int32)
    if per_row_lengths:
        seq_lens = torch.tensor(row_ends, device=device, dtype=torch.int32).view(
            -1, next_n
        )
    else:
        seq_lens = torch.tensor(lengths, device=device, dtype=torch.int32)

    topk = SparseIndexerTopk("per_row")
    topk(logits, seq_lens, next_n, indices, top_k, max(lengths))

    _assert_topk_matches_reference(logits, indices, [0] * len(row_ends), row_ends)


@pytest.mark.parametrize("next_n", [1, 3])
@pytest.mark.parametrize("per_row_lengths", [False, True])
@torch.inference_mode()
def test_top_k_per_row_decode(device, next_n, per_row_lengths):
    _run_top_k_per_row_decode_test(
        device, [next_n, 3072], next_n, per_row_lengths, 512, 4096
    )


@torch.inference_mode()
def test_top_k_per_row_decode_large_vocab_size(device):
    _run_top_k_per_row_decode_test(device, [250001, 299891], 2, False, 2048, 300000)


@pytest.mark.parametrize(
    "op_name",
    [
        pytest.param(
            "persistent_topk",
            marks=pytest.mark.xfail(
                raises=AssertionError,
                strict=True,
                reason="MACA persistent Top-K returns incorrect indices for mixed lengths",
            ),
        ),
        "cooperative_topk",
    ],
)
@pytest.mark.parametrize("top_k", [512, 1024, 2048])
def test_workspace_topk_padded_stride(device, op_name, top_k):
    MacaPlatform.import_kernels()
    row_ends = [7, 2051, 3013, 18]
    logits = _make_logits(device, len(row_ends), 4112)[:, :4096]
    lengths = torch.tensor(row_ends, dtype=torch.int32, device=device)
    indices = torch.full((len(row_ends), top_k), -2, dtype=torch.int32, device=device)
    workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device=device)

    getattr(torch.ops._C, op_name)(
        logits, lengths, indices, workspace, top_k, max(row_ends)
    )

    _assert_topk_matches_reference(logits, indices, [0] * len(row_ends), row_ends)
