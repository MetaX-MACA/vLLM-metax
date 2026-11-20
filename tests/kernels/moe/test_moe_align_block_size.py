# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from collections import defaultdict

import pytest
import torch

from vllm_metax.platform import MacaPlatform

pytestmark = pytest.mark.requires_maca


# Adapted from vLLM v0.33.0 tests/kernels/moe/test_moe_align_block_size.py.
@pytest.mark.parametrize(
    ("experts", "tokens", "block"), [(32, 8, 32), (256, 8, 32), (257, 1024, 128)]
)
@pytest.mark.parametrize("mapped", [False, True])
def test_moe_align_places_each_valid_route_in_its_expert_blocks(
    device, experts, tokens, block, mapped
):
    MacaPlatform.import_kernels()
    ids = (
        torch.arange(tokens * 4, device=device, dtype=torch.int32).reshape(tokens, 4)
        % experts
    )
    ids[0] = torch.tensor([-1, experts, 0, 1], device=device, dtype=ids.dtype)
    expert_map = None
    if mapped:
        expert_map = torch.full((experts,), -1, device=device, dtype=torch.int32)
        expert_map[::2] = torch.arange(
            (experts + 1) // 2, device=device, dtype=torch.int32
        )

    capacity = ids.numel() + experts * (block - 1)
    sorted_ids = torch.empty(capacity, device=device, dtype=torch.int32)
    expert_ids = torch.empty(
        (capacity + block - 1) // block, device=device, dtype=torch.int32
    )
    padded = torch.empty(1, device=device, dtype=torch.int32)
    scatter = torch.empty((ids.numel(), 1), device=device, dtype=torch.int32)
    torch.ops._moe_C.moe_align_block_size(
        ids, experts, block, sorted_ids, expert_ids, padded, expert_map, scatter
    )

    by_expert = defaultdict(list)
    expected_scatter = []
    for route, expert in enumerate(ids.flatten().tolist()):
        valid = 0 <= expert < experts and (not mapped or expert % 2 == 0)
        expected_scatter.append(route if valid else -1)
        if valid:
            by_expert[expert // 2 if mapped else expert].append(route)
    expected_blocks = [
        expert
        for expert in sorted(by_expert)
        for _ in range((len(by_expert[expert]) + block - 1) // block)
    ]
    assert padded.item() == len(expected_blocks) * block
    assert expert_ids[: len(expected_blocks)].tolist() == expected_blocks
    assert scatter.flatten().tolist() == expected_scatter

    actual_by_expert = defaultdict(list)
    for expert, routes in zip(
        expected_blocks,
        sorted_ids[: padded.item()].view(-1, block).tolist(),
        strict=True,
    ):
        assert all(0 <= route <= ids.numel() for route in routes)
        actual_by_expert[expert].extend(
            route for route in routes if route < ids.numel()
        )
    assert {
        expert: sorted(routes) for expert, routes in actual_by_expert.items()
    } == by_expert


@pytest.mark.parametrize("max_tokens", [13, 512])
@pytest.mark.parametrize("block_size", [8, 32, 48])
@pytest.mark.parametrize("all_empty", [False, True])
def test_batched_moe_align_block_size(device, max_tokens, block_size, all_empty):
    MacaPlatform.import_kernels()
    counts = [max_tokens, 0, 5, 1, 0, 2, max_tokens - 1, 0]
    if all_empty:
        counts = [0] * len(counts)
    experts = len(counts)
    max_blocks = (max_tokens + block_size - 1) // block_size * experts
    counts_tensor = torch.tensor(counts, device=device, dtype=torch.int32)
    tokens = torch.empty(max_blocks * block_size, device=device, dtype=torch.int32)
    expert_ids = torch.empty(max_blocks, device=device, dtype=torch.int32)
    padded = torch.empty(1, device=device, dtype=torch.int32)

    torch.ops._moe_C.batched_moe_align_block_size(
        max_tokens, block_size, counts_tensor, tokens, expert_ids, padded
    )

    sentinel = max_tokens * experts
    expected_tokens, expected_experts = [], []
    for expert, count in enumerate(counts):
        expert_tokens = list(range(expert * max_tokens, expert * max_tokens + count))
        expected_tokens.extend(expert_tokens + [sentinel] * (-count % block_size))
        expected_experts.extend([expert] * ((count + block_size - 1) // block_size))
    assert padded.item() == len(expected_tokens)
    assert tokens.tolist() == expected_tokens + [sentinel] * (
        tokens.numel() - len(expected_tokens)
    )
    assert expert_ids.tolist() == expected_experts + [-1] * (
        max_blocks - len(expected_experts)
    )


@pytest.mark.parametrize("disable_adapter", [False, True])
def test_moe_lora_align_block_size(device, disable_adapter):
    MacaPlatform.import_kernels()
    routes = [[1, 2], [3, 0], [3, 2], [2, 1], [5, 4], [0, 1]]
    token_lora = [0, 1, -1, 0, 1, 0]
    num_experts, block_size, max_loras = 8, 4, 2
    num_routes = len(routes) * len(routes[0])
    max_tokens = (
        (num_routes + num_experts * (block_size - 1) + block_size - 1)
        // block_size
        * block_size
    )
    max_blocks = max_tokens // block_size

    def i32(values):
        return torch.tensor(values, dtype=torch.int32, device=device)

    ordered = torch.full((max_loras, max_tokens), -7, dtype=torch.int32, device=device)
    expert_ids = torch.full(
        (max_loras, max_blocks), -2, dtype=torch.int32, device=device
    )
    padded = torch.full((max_loras,), -13, dtype=torch.int32, device=device)
    torch.ops._moe_C.moe_lora_align_block_size(
        i32(routes),
        i32(token_lora),
        num_experts,
        block_size,
        max_loras,
        max_tokens,
        max_blocks,
        ordered.flatten(),
        expert_ids.flatten(),
        padded,
        i32([1, int(not disable_adapter), 1]),
        i32([-1, 0, 1]),
        None,
    )

    for lora in range(max_loras):
        if lora == 1 and disable_adapter:
            assert padded[lora].item() == -13
            assert (expert_ids[lora] == -2).all()
            assert (ordered[lora] == -7).all()
            continue
        expected = defaultdict(list)
        for token, pair in enumerate(routes):
            if token_lora[token] == lora:
                for route, expert in enumerate(pair):
                    expected[expert].append(token * len(pair) + route)
        size = padded[lora].item()
        assert size == sum(
            (len(ids) + block_size - 1) // block_size * block_size
            for ids in expected.values()
        )
        actual = defaultdict(list)
        for block, expert in zip(
            ordered[lora, :size].view(-1, block_size).tolist(),
            expert_ids[lora, : size // block_size].tolist(),
            strict=True,
        ):
            actual[expert].extend(i for i in block if i != num_routes)
        assert {e: sorted(ids) for e, ids in actual.items()} == expected
