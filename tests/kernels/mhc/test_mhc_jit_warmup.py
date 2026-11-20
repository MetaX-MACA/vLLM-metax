# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest

from vllm_metax.model_executor.kernels.mhc.warmup import MHC_PRE_NORM_KERNEL


@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize(
    ("split_mode", "write_aux"), [("fused", False), ("input", True)]
)
def test_tilelang_warmup_inputs_reproduce_compile_key(
    hidden_size, split_mode, write_aux
):
    kernel = MHC_PRE_NORM_KERNEL
    fields = dict(
        hidden_size=hidden_size,
        rms_eps=1e-20,
        hc_pre_eps=1e-6,
        hc_sinkhorn_eps=1e-6,
        hc_post_mult_value=2.0,
        sinkhorn_repeat=20,
        norm_eps=1e-6,
        hc_mult=4,
        use_pre_mix_in=False,
        rms_numel=4 * hidden_size,
        split_mode=split_mode,
        write_aux=write_aux,
    )
    extra_splits = [1, 4, 8]
    keys = kernel.get_warmup_keys(max_tokens=129, extra_splits=extra_splits, **fields)

    assert len(keys) == len(set(keys))
    assert {kernel.dispatch(n_splits=n, **fields) for n in extra_splits} <= set(keys)
    assert all(key == kernel.dispatch(n_splits=key.n_splits, **fields) for key in keys)
