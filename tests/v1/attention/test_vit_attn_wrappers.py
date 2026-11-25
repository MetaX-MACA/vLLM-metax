# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm_metax.v1.attention.ops.vit_attn_wrappers import vit_flash_attn_wrapper

pytestmark = pytest.mark.requires_maca


# Adapted from vLLM v0.33.0 tests/v1/attention/test_vit_attn_wrappers.py.
def test_flash_vit_fake_op_preserves_shape_and_contiguous_layout():
    with FakeTensorMode():
        x = torch.empty((2, 3, 5, 64), dtype=torch.bfloat16).transpose(1, 2)
        output = vit_flash_attn_wrapper(x, x, x, 2, False, 2)
        assert output.shape == x.shape
        assert output.dtype == x.dtype
        assert output.device == x.device
        assert output.is_contiguous()


@pytest.mark.parametrize("graph_padding", [False, True])
@pytest.mark.parametrize("head_size", [64, 72])
@torch.inference_mode()
def test_flash_vit_registered_op_matches_sdpa(device, graph_padding, head_size):
    torch.manual_seed(0)
    q, k, v = [
        torch.randn(2, 3, 8, head_size, device=device, dtype=torch.bfloat16).transpose(1, 2)
        for _ in range(3)
    ]
    lengths = [5, 7] if graph_padding else [8, 8]
    cu_seqlens = torch.tensor(
        [0, lengths[0], sum(lengths)], device=device, dtype=torch.int32
    )
    max_seqlen = torch.tensor(max(lengths), device=device, dtype=torch.int32)
    output = vit_flash_attn_wrapper(
        q,
        k,
        v,
        2,
        False,
        2,
        cu_seqlens=cu_seqlens if graph_padding else None,
        max_seqlen=max_seqlen if graph_padding else None,
    )
    assert output.is_contiguous()
    expected = torch.zeros_like(q).contiguous().flatten(0, 1)
    start = 0
    for length in lengths:
        parts = [
            x.flatten(0, 1)[start : start + length].transpose(0, 1).float()
            for x in (q, k, v)
        ]
        expected[start : start + length] = (
            torch.nn.functional.scaled_dot_product_attention(
                *parts,
            )
            .transpose(0, 1)
            .to(q.dtype)
        )
        start += length
    torch.testing.assert_close(
        output, expected.reshape(q.shape), atol=1.5e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        output.flatten(0, 1)[start:], expected[start:], atol=0, rtol=0
    )
