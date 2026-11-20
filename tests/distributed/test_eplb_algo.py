# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch

from vllm.distributed.eplb.policy.default import DefaultEplbPolicy

from vllm_metax.patch.enhancement.static_eplb.placement import (
    map_from_payload,
    read_eplb_file,
    select_model_payload,
    write_eplb_file,
)


# Adapted from the placement invariants in vLLM v0.33.0
# tests/distributed/test_eplb_algo.py.
@pytest.mark.parametrize("format", ["json", "pt"])
def test_exported_eplb_counts_rebalance_the_selected_model(tmp_path, format):
    counts = torch.tensor(
        [
            [[4, 12, 1, 55], [32, 1, 14, 6]],
            [[8, 6, 40, 3], [0, 8, 11, 2]],
        ]
    )
    old_map = torch.tensor([[0, 1, 2, 3, 1, 3], [0, 1, 2, 3, 0, 2]])
    path = tmp_path / f"expert-placement.{format}"
    write_eplb_file(
        path,
        {
            "models": [
                {"model_name": "target", "logical_count": counts.flip(-1)},
                {"model_name": "draft", "logical_count": counts, "ep_size": 2},
            ],
        },
    )
    payload = select_model_payload(
        read_eplb_file(path),
        model_name="draft",
        model_index=1,
        model_count=2,
    )
    policy = DefaultEplbPolicy()
    actual = map_from_payload(
        payload,
        num_layers=2,
        num_physical_experts=6,
        num_logical_experts=4,
        ep_size=2,
        num_groups=1,
        num_nodes=1,
        old_map=old_map,
        policy=policy,
    )
    expected = policy.rebalance_experts(counts.sum(dim=0), 6, 1, 1, 2, old_map)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    ("mapping", "error"),
    [
        ([[0, 1, 2, 3]], "shape"),
        ([[0, 1, 2, 4, 0, 1]], "invalid expert ID"),
        ([[0, 1, 2, 1, 0, 1]], "must have a physical slot"),
        ([[0.5, 1, 2, 3, 0, 1]], "must contain integers"),
    ],
)
def test_eplb_import_rejects_invalid_placements(mapping, error):
    old_map = torch.tensor([[0, 1, 2, 3, 0, 1]])
    with pytest.raises(ValueError, match=error):
        map_from_payload(
            {"physical_to_logical_map": mapping},
            num_layers=1,
            num_physical_experts=6,
            num_logical_experts=4,
            ep_size=2,
            num_groups=1,
            num_nodes=1,
            old_map=old_map,
            policy=DefaultEplbPolicy(),
        )
