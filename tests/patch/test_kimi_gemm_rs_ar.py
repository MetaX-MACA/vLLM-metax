# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

import inspect
import sys

import pytest

from vllm.plugins import load_general_plugins

load_general_plugins()

from vllm.models.kimi_k3.nvidia import model
from vllm_metax.patch.models.kimi_k3 import disable_gemm_rs_ar


@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("ar_enabled,rs_enabled", [("0", "0"), ("1", "0"), ("0", "1"), ("1", "1")])
def test_initializer_skips_cuda_only_import(monkeypatch, sequence_parallel, ar_enabled, rs_enabled):
    monkeypatch.setenv("VLLM_KIMI_K3_GEMM_AR", ar_enabled)
    monkeypatch.setenv("VLLM_ENABLE_GEMM_RS", rs_enabled)
    monkeypatch.setitem(sys.modules, "vllm.model_executor.kernels.linear.cute_dsl.gemm_rs_ar", None)
    assert model.maybe_init_gemm_rs_ar is disable_gemm_rs_ar._maybe_init_gemm_rs_ar
    assert model.maybe_init_gemm_rs_ar(None, use_sequence_parallel=sequence_parallel) is False


def test_initializer_preserves_target_parameter_names():
    assert list(inspect.signature(model.maybe_init_gemm_rs_ar).parameters) == [
        "vllm_config", "use_sequence_parallel"
    ]
