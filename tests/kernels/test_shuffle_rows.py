# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch

from vllm_metax.platform import MacaPlatform

pytestmark = pytest.mark.requires_maca


def _shuffle_rows(x, indices):
    MacaPlatform.import_kernels()
    out = x.new_empty((indices.numel(), x.size(1)))
    torch.ops._moe_C.shuffle_rows(x, indices, out)
    assert out.device == x.device
    assert out.dtype == x.dtype
    assert out.is_contiguous()
    return out


@pytest.mark.parametrize("tokens", [1, 16, 64, 128, 256, 512, 1024])
@pytest.mark.parametrize("hidden", [128, 256, 512, 1024, 2048, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_shuffle_rows_basic(device, tokens, hidden, dtype):
    x = torch.randn(tokens, hidden, device=device, dtype=dtype)
    indices = torch.arange(tokens, device=device, dtype=torch.int32)
    torch.testing.assert_close(_shuffle_rows(x, indices), x, atol=0, rtol=0)


@pytest.mark.parametrize("mode", ["reverse", "random", "duplicate", "expand"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_shuffle_rows_permutation(device, mode, dtype):
    tokens, hidden = 32, 512
    rng = torch.Generator(device=device).manual_seed(42)
    x = torch.randn(tokens, hidden, device=device, dtype=dtype, generator=rng)
    ids = torch.arange(tokens, device=device, dtype=torch.int32)
    if mode == "reverse":
        ids = ids.flip(0)
    elif mode == "random":
        ids = torch.randperm(tokens, device=device, dtype=torch.int32, generator=rng)
    elif mode == "duplicate":
        ids = ids.repeat_interleave(2)
    else:
        ids = torch.randint(
            tokens, (2 * tokens,), device=device, dtype=torch.int32, generator=rng
        )
    torch.testing.assert_close(_shuffle_rows(x, ids), x[ids.long()], atol=0, rtol=0)


def test_shuffle_rows_invalid_indices_are_zeroed(device):
    x = torch.randn(3, 128, device=device, dtype=torch.bfloat16)
    indices = torch.tensor([-1, 1, 3], device=device, dtype=torch.int32)
    expected = torch.stack([torch.zeros_like(x[0]), x[1], torch.zeros_like(x[0])])
    torch.testing.assert_close(_shuffle_rows(x, indices), expected, atol=0, rtol=0)


def test_shuffle_rows_edge_cases(device):
    x = torch.randn(1, 8, device=device, dtype=torch.float16)
    index = torch.zeros(1, device=device, dtype=torch.int32)
    torch.testing.assert_close(_shuffle_rows(x, index), x, atol=0, rtol=0)

    with pytest.raises(RuntimeError, match="num_cols must be divisible"):
        _shuffle_rows(x[:, :1].contiguous(), index)
