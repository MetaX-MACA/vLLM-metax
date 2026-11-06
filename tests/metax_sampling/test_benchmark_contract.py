# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Keep whole-step timing paths distinct without needing GPU execution."""

import torch

from benchmarks.benchmark_windowed_penalties import update_and_apply
from vllm_metax.v1.sample.windowed_penalties import (
    SparsePenaltyBatch,
    WindowedPenaltyConfig,
    WindowedPenaltyState,
)


def test_exact_kernel_callback_observes_rebuilt_state():
    batch = SparsePenaltyBatch(torch.device("cpu"))
    request = {
        0: WindowedPenaltyState(
            WindowedPenaltyConfig.parse({"window": 1, "frequency": 1}), [2]
        )
    }
    observations = []

    def apply(logits):
        observations.append(batch.indices.tolist())
        return batch.apply_torch(logits)

    assert update_and_apply(
        batch, request, torch.ones(1, 4), apply_fn=apply
    ).tolist() == [[1, 1, 0, 1]]
    request[0].output_token_ids[:] = [3]
    assert update_and_apply(
        batch, request, torch.ones(1, 4), apply_fn=apply
    ).tolist() == [[1, 1, 1, 0]]
    assert observations == [[[0], [2]], [[0], [3]]]


def test_sparse_torch_reference_and_default_paths_are_equivalent():
    batch = SparsePenaltyBatch(torch.device("cpu"))
    request = {
        0: WindowedPenaltyState(
            WindowedPenaltyConfig.parse({"window": 2, "repetition": 2}), [1, 1]
        )
    }
    logits = torch.tensor([[0.0, 8.0, -8.0]])
    expected = torch.tensor([[0.0, 4.0, -8.0]])
    torch.testing.assert_close(
        update_and_apply(batch, request, logits.clone()), expected
    )
    torch.testing.assert_close(
        update_and_apply(batch, request, logits.clone(), torch_reference=True), expected
    )
