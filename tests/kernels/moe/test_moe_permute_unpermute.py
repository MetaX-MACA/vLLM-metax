# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch

from vllm.model_executor.layers.fused_moe.moe_permute_unpermute import (
    MoEPermuteScratch,
    moe_permute,
    moe_permute_unpermute_supported,
    moe_prepare_scatter,
    moe_unpermute,
)

from vllm_metax.platform import MacaPlatform

pytestmark = pytest.mark.requires_maca


@pytest.mark.parametrize("use_scratch", [False, True])
@pytest.mark.parametrize("expert_parallel", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_moe_permute_unpermute(device, dtype, expert_parallel, use_scratch):
    MacaPlatform.import_kernels()
    assert moe_permute_unpermute_supported()
    n_expert, top_k = 4, 2
    local_experts = 2 if expert_parallel else n_expert
    expert_map = (
        torch.tensor([-1, 0, -1, 1], device=device, dtype=torch.int32)
        if expert_parallel
        else None
    )
    x = torch.arange(5 * 32, dtype=dtype, device=device).view(5, 32) / 32
    assignments = torch.tensor(
        [[3, 1], [0, 3], [1, 2], [2, 0], [1, 3]], dtype=torch.int32, device=device
    )
    weights = torch.tensor(
        [[0.25, 0.75], [0.125, 0.875], [0.25, 0.75], [0.5, 0.5], [0.625, 0.375]],
        device=device,
    )
    local_ids = assignments if expert_map is None else expert_map[assignments.long()]
    scratch = (
        MoEPermuteScratch(
            num_experts=n_expert, num_local_experts=local_experts, device=device
        )
        if use_scratch
        else None
    )

    permuted, _, offsets, inverse, _ = moe_permute(
        x, None, assignments, n_expert, local_experts, expert_map, scratch=scratch
    )

    valid = local_ids.flatten() >= 0
    source_rows = torch.where(valid)[0]
    local = local_ids.flatten()[valid]
    order = local.argsort(stable=True)
    source_rows = source_rows[order]
    expected_offsets = torch.nn.functional.pad(
        local.bincount(minlength=local_experts).cumsum(0), (1, 0)
    )
    torch.testing.assert_close(offsets, expected_offsets)
    torch.testing.assert_close(
        permuted[: len(source_rows)], x[source_rows // top_k], atol=0, rtol=0
    )
    torch.testing.assert_close(
        inverse[source_rows],
        torch.arange(len(source_rows), device=device, dtype=inverse.dtype),
    )
    assert (inverse[~valid] >= len(source_rows)).all()

    payload = torch.zeros_like(permuted)
    payload[: len(source_rows)] = permuted[: len(source_rows)] + local[order, None]
    result = torch.empty_like(x)
    moe_unpermute(result, payload, weights, inverse, offsets)
    expected = (
        (x.float()[:, None] + local_ids[..., None])
        * (weights * (local_ids >= 0))[..., None]
    ).sum(dim=1)
    torch.testing.assert_close(result, expected.to(dtype), atol=1e-2, rtol=0)
    if scratch is not None:
        scatter_offsets, scatter = moe_prepare_scatter(assignments, expert_map, scratch)
        torch.testing.assert_close(scatter_offsets, expected_offsets)
        torch.testing.assert_close(scatter.flatten()[source_rows], inverse[source_rows])
