# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import gc

import pytest
import torch

from vllm_metax.platform import MacaPlatform

pytestmark = pytest.mark.requires_maca


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_permute_cols(device, dtype):
    MacaPlatform.import_kernels()
    rng = torch.Generator(device=device).manual_seed(17)
    x = torch.randn(7, 9, 272, device=device, dtype=dtype, generator=rng)
    permutation = torch.randperm(
        x.size(-1), device=device, dtype=torch.int32, generator=rng
    )

    actual = torch.ops._C.permute_cols(x, permutation)

    torch.testing.assert_close(actual, x.index_select(-1, permutation), atol=0, rtol=0)
    with pytest.raises(RuntimeError, match="columns must be a multiple of 8"):
        torch.ops._C.permute_cols(x[..., :-1].contiguous(), permutation[:-1])


def test_weak_ref_tensor_alias_preserves_strides(device):
    MacaPlatform.import_kernels()
    source = torch.arange(48, device=device, dtype=torch.float32).view(6, 8)
    original = source[1::2, 1::2].T

    view = torch.ops._C.weak_ref_tensor(original)

    assert view.data_ptr() == original.data_ptr()
    assert view.stride() == original.stride()
    assert view.device == original.device
    view[1, 0] = -7
    assert source[1, 3].item() == -7
    torch.testing.assert_close(view, original, atol=0, rtol=0)


@pytest.mark.parametrize("pinned", [False, True])
def test_get_cuda_view_from_cpu_tensor(device, pinned):
    MacaPlatform.import_kernels()
    host = torch.empty((6, 16), dtype=torch.int32, pin_memory=pinned)
    host.copy_(torch.arange(host.numel(), dtype=torch.int32).view_as(host))
    original = host.T[::2]
    expected = original.clone()

    alias = torch.ops._C.get_cuda_view_from_cpu_tensor(original)

    assert alias.device.type == device.type
    torch.testing.assert_close(alias.cpu(), expected, atol=0, rtol=0)
    if pinned:
        assert alias.stride() == original.stride()
        original[0, 0] = 123
        expected[0, 0] = 123
        assert alias[0, 0].item() == 123
    del original, host
    gc.collect()
    torch.testing.assert_close(alias.cpu(), expected, atol=0, rtol=0)
