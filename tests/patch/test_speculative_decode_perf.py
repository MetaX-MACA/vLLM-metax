# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest


@pytest.fixture
def reorder_fn(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.worker.gpu_model_runner",
        types.SimpleNamespace(GPUModelRunner=type("GPUModelRunner", (), {})),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm.v1.attention.backends.utils",
        types.SimpleNamespace(
            reorder_batch_to_split_decodes_and_prefills=lambda *args, **kwargs: None
        ),
    )

    module_path = (
        Path(__file__).parents[2]
        / "vllm_metax"
        / "patch"
        / "performance"
        / "speculative_decode_perf.py"
    )
    spec = importlib.util.spec_from_file_location(
        "speculative_decode_perf_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module._metax_reorder_batch_to_split_decodes_and_prefills


class FakeInputBatch:
    def __init__(self, req_ids: list[str], computed_tokens: list[int]):
        self.req_ids = req_ids
        self.num_computed_tokens_cpu = np.array(computed_tokens, dtype=np.int32)
        self.swaps: list[tuple[int, int]] = []

    def swap_states(self, src: int, dst: int) -> None:
        self.swaps.append((src, dst))
        self.req_ids[src], self.req_ids[dst] = self.req_ids[dst], self.req_ids[src]
        self.num_computed_tokens_cpu[src], self.num_computed_tokens_cpu[dst] = (
            self.num_computed_tokens_cpu[dst],
            self.num_computed_tokens_cpu[src],
        )


@dataclass
class FakeSchedulerOutput:
    num_scheduled_tokens: dict[str, int]


def test_metax_reorder_sorts_decode_then_extend_then_prefill(reorder_fn):
    batch = FakeInputBatch(
        req_ids=["ext4", "pre4", "dec2", "dec1", "ext5", "pre1", "dec3", "pre2"],
        computed_tokens=[8, 0, 9, 7, 6, 0, 5, 0],
    )
    scheduler_output = FakeSchedulerOutput(
        num_scheduled_tokens={
            "ext4": 4,
            "pre4": 4,
            "dec2": 2,
            "dec1": 1,
            "ext5": 5,
            "pre1": 1,
            "dec3": 3,
            "pre2": 2,
        }
    )

    changed = reorder_fn(batch, scheduler_output, decode_threshold=3)

    assert changed is True
    assert batch.req_ids == [
        "dec1",
        "dec2",
        "dec3",
        "ext4",
        "ext5",
        "pre4",
        "pre1",
        "pre2",
    ]
    assert batch.swaps


def test_metax_reorder_keeps_decode_sort_stable_for_equal_lengths(reorder_fn):
    batch = FakeInputBatch(
        req_ids=["dec_a", "dec_b", "ext", "pre"],
        computed_tokens=[3, 4, 5, 0],
    )
    scheduler_output = FakeSchedulerOutput(
        num_scheduled_tokens={"dec_a": 2, "dec_b": 2, "ext": 4, "pre": 1}
    )

    changed = reorder_fn(batch, scheduler_output, decode_threshold=2)

    assert changed is False
    assert batch.req_ids == ["dec_a", "dec_b", "ext", "pre"]
    assert batch.swaps == []


def test_metax_reorder_returns_false_when_order_is_already_target(reorder_fn):
    batch = FakeInputBatch(
        req_ids=["dec1", "dec2", "ext", "pre"],
        computed_tokens=[7, 8, 9, 0],
    )
    scheduler_output = FakeSchedulerOutput(
        num_scheduled_tokens={"dec1": 1, "dec2": 2, "ext": 8, "pre": 1}
    )

    changed = reorder_fn(batch, scheduler_output, decode_threshold=2)

    assert changed is False
    assert batch.req_ids == ["dec1", "dec2", "ext", "pre"]
    assert batch.swaps == []
