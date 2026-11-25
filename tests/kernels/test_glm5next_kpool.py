# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

import pytest
import torch

from vllm_metax.models.glm5next.metax.kpool_bf16 import (
    kpool_decode_update_and_maybe_write_cache_bf16,
)
from vllm_metax.models.glm5next.metax.kpool_int8 import (
    kpool_decode_update_and_maybe_write_cache_int8,
)


@pytest.mark.requires_maca
@pytest.mark.parametrize("cache_format", ["bf16", "int8"])
def test_kpool_decode_replays_pool_at_speculative_tail_boundary(device, cache_format):
    head_dim = 128
    pool_size = 4
    ring_size = 8
    tail = torch.zeros((2, 2, ring_size, head_dim), device=device, dtype=torch.bfloat16)
    ape = torch.zeros((pool_size, head_dim), device=device, dtype=torch.float32)
    if cache_format == "bf16":
        update = kpool_decode_update_and_maybe_write_cache_bf16
        cache = torch.zeros((1, 2, head_dim), device=device, dtype=torch.bfloat16)
    else:
        update = kpool_decode_update_and_maybe_write_cache_int8
        cache = torch.zeros((1, 2, head_dim + 4), device=device, dtype=torch.uint8)

    def decode(values, positions, pool_slots):
        key = torch.tensor(values, device=device, dtype=torch.bfloat16)
        key = key[None, :, None].expand(1, -1, head_dim).contiguous()
        positions = torch.tensor([positions], device=device, dtype=torch.int32)
        tail_slots = ring_size + positions % ring_size
        update(
            cache,
            tail,
            tail_slots,
            key,
            torch.zeros_like(key),
            ape,
            torch.tensor([pool_slots], device=device, dtype=torch.int32),
            positions,
            pool_size,
            head_dim,
        )

    decode([1, 2, 3, 4, 100, 200], [4, 5, 6, 7, 8, 9], [-1, -1, -1, 0, -1, -1])
    # A rejected draft at position 7 is redone after drafts at 8 and 9 were stashed.
    decode([14], [7], [0])
    assert tail[1, 0, :, 0].tolist() == [100, 200, 0, 0, 1, 2, 3, 14]
    assert tail[0].count_nonzero().item() == 0
    if cache_format == "bf16":
        actual = cache[0, 0].float()
    else:
        scale = cache.view(torch.float32).flatten()[2 * head_dim // 4]
        actual = cache.view(torch.int8).flatten()[:head_dim].float() * scale
    expected = torch.zeros(head_dim, device=device)
    expected[0] = 5 * head_dim**0.5
    torch.testing.assert_close(actual, expected, atol=0.25, rtol=0)
