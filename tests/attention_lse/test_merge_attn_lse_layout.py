# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Target-runtime tests, separate from the isolated HPCC experiment."""

import importlib

import pytest
import torch

from vllm_metax import envs

ops = importlib.import_module("vllm_metax.v1.attention.ops.merge_attn_states")


@pytest.mark.parametrize("maca", [False, True])
@pytest.mark.parametrize("precompiled", [False, True])
def test_prepare_lse_provider_contract(monkeypatch, maca, precompiled):
    # This test simulates only the provider decision, not a working MACA kernel.
    monkeypatch.setattr(ops.current_platform, "is_maca", lambda: maca)
    monkeypatch.setattr(envs, "USE_PRECOMPILED_KERNEL", precompiled)
    base = torch.arange(120).reshape(10, 12)
    for view in (base, base.t(), base[1::2, 1::3].t(), base[:0].t()):
        out = ops.prepare_merge_attn_lse(view)
        torch.testing.assert_close(out, view)
        if maca and precompiled:
            assert out.is_contiguous()
        else:
            assert out is view
            assert out.stride() == view.stride()
            assert out.storage_offset() == view.storage_offset()


@pytest.mark.parametrize(
    "device",
    [
        pytest.param(
            "cpu",
            marks=pytest.mark.skipif(
                not ops.current_platform.is_cpu(), reason="CPU platform required"
            ),
        ),
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="GPU runtime required"
            ),
        ),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1, 1, 64), (8, 4, 80), (33, 8, 128)])
@pytest.mark.parametrize("context", [None, 0, 1])
@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("base2", [False, True])
def test_merge_lse_layout_equivalence(device, dtype, shape, context, empty, base2):
    torch.manual_seed(1234)
    n, h, d = shape
    p = torch.randn(n, h, d, dtype=dtype, device=device)
    s = torch.randn_like(p)
    lp = torch.randn(n * 2 + 2, h * 2 + 2, device=device)[
        1 : 1 + n * 2 : 2, 1 : 1 + h * 2 : 2
    ].t()
    ls = torch.randn_like(lp)
    if empty:
        lp.fill_(-float("inf"))
        ls.fill_(-float("inf"))
        # Native merge specifies that all-empty prefix outputs are zero.
        p.zero_()
        s.zero_()
    lp_ref, ls_ref = lp.contiguous(), ls.contiguous()
    lp_new, ls_new = ops.prepare_merge_attn_lse(lp), ops.prepare_merge_attn_lse(ls)
    if base2:
        lp_ref, ls_ref = lp_ref * 0.6931471805599453, ls_ref * 0.6931471805599453
        lp_new, ls_new = lp_new * 0.6931471805599453, ls_new * 0.6931471805599453
    baseline, candidate = torch.empty_like(p), torch.empty_like(p)
    lse_ref = torch.empty(h, n, device=device)
    lse_new = torch.empty_like(lse_ref)
    ops.merge_attn_states(baseline, p, lp_ref, s, ls_ref, lse_ref, context)
    ops.merge_attn_states(candidate, p, lp_new, s, ls_new, lse_new, context)
    if device == "cuda":
        torch.cuda.synchronize()
    torch.testing.assert_close(candidate, baseline)
    torch.testing.assert_close(lse_new, lse_ref)
