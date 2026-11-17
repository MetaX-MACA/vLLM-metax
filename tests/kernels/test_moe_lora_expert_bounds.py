# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

import hashlib
import json
import os
from pathlib import Path
import sys

import pytest
import torch

from vllm.plugins import load_general_plugins

load_general_plugins()

from vllm.lora.ops.triton_ops import fused_moe_lora_op as upstream
from vllm_metax.patch.bugfix import fused_moe_lora_expert_bounds as bounds


@pytest.fixture(scope="module")
def numerical_results():
    results = []
    yield results
    if destination := os.environ.get("BRANCH_PORT_NUMERIC_RESULTS"):
        Path(destination).write_text(json.dumps({
            "cases": results,
            "wrapper": str(Path(bounds.__file__)),
            "wrapper_sha256": hashlib.sha256(Path(bounds.__file__).read_bytes()).hexdigest(),
            "kernel_type": type(upstream._fused_moe_lora_small_batch_kernel).__name__,
            "provider": "source", "use_precompiled_kernel": os.environ.get("USE_PRECOMPILED_KERNEL"),
            "loaded_mcoplib_modules": [name for name in sys.modules if name == "mcoplib" or name.startswith("mcoplib.")],
        }, indent=2) + "\n")


def _case(results, *, dtype, slices, add_inputs, routed, tokens=6):
    assert upstream._run_fused_moe_lora_small_batch is bounds._run_fused_moe_lora_small_batch
    assert upstream._fused_moe_lora_small_batch_kernel is bounds._fused_moe_lora_small_batch_kernel
    generator = torch.Generator().manual_seed(0)
    E, rank, K, N, top_k = 3, 4, 33, 129, 3
    lora_map = torch.tensor(([0, 1, 0, -1, 2, 0] * ((tokens + 5) // 6))[:tokens], dtype=torch.int32)
    experts = torch.tensor(([-1, 0, E, 0, 1, 2, 2, E + 4, 1, 0, 1, 2, 0, 1, 2, 1, 2, -1] * ((tokens + 5) // 6))[:tokens * top_k], dtype=torch.int32).reshape(tokens, top_k)
    enabled = torch.tensor([1, 0], dtype=torch.int32)
    a = [(torch.randn(2, E, rank, K, generator=generator) * 0.1).to(dtype) for _ in range(slices)]
    b = [(torch.randn(2, E, N, rank, generator=generator) * 0.1).to(dtype) for _ in range(slices)]
    x = (torch.randn(tokens * top_k if routed else tokens, K, generator=generator) * 0.1).to(dtype)
    weights = torch.rand(tokens, top_k, generator=generator)
    initial = (torch.randn(tokens, top_k, slices * N, generator=generator) * 0.1).to(dtype)
    expected = initial.double().clone()
    valid = torch.zeros(tokens, top_k, dtype=torch.bool)
    for token in range(tokens):
        adapter = int(lora_map[token])
        for top in range(top_k):
            expert = int(experts[token, top])
            if not (0 <= adapter < 2 and enabled[adapter] and 0 <= expert < E):
                continue
            valid[token, top] = True
            row = token * top_k + top if routed else token
            for slice_id in range(slices):
                delta = b[slice_id][adapter, expert].double() @ (a[slice_id][adapter, expert].double() @ x[row].double())
                if routed:
                    delta *= weights[token, top].double()
                part = slice(slice_id * N, (slice_id + 1) * N)
                expected[token, top, part] = initial[token, top, part].double() + delta if add_inputs else delta
    actual = initial.cuda()
    upstream._run_fused_moe_lora_small_batch(actual, x.cuda(), [v.cuda() for v in a], [v.cuda() for v in b], weights.cuda(), experts.flatten().cuda(), lora_map.cuda(), top_k, enabled.cuda(), routed, add_inputs)
    actual = actual.cpu()
    assert torch.equal(actual[~valid], initial[~valid])
    assert torch.isfinite(actual).all()
    atol, rtol = (0.002, 0.02) if dtype == torch.bfloat16 else (0.0002, 0.005)
    torch.testing.assert_close(actual.double(), expected, atol=atol, rtol=rtol)
    results.append({"id": f"{dtype}-slices{slices}-add{add_inputs}-routed{routed}-tokens{tokens}", "max_abs": (actual.double() - expected).abs().max().item(), "reference_scale": expected.abs().max().item(), "atol": atol, "rtol": rtol, "nan_count": 0, "inf_count": 0, "passed": True, "invalid_output_unchanged": True, "valid_rows": int(valid.sum())})


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("slices", [1, 2])
@pytest.mark.parametrize("add_inputs", [False, True])
@pytest.mark.parametrize("routed", [False, True])
def test_expert_and_adapter_bounds(numerical_results, dtype, slices, add_inputs, routed):
    _case(numerical_results, dtype=dtype, slices=slices, add_inputs=add_inputs, routed=routed)


def test_persistent_loop_keeps_work_after_invalid_entries(numerical_results):
    _case(numerical_results, dtype=torch.bfloat16, slices=2, add_inputs=True, routed=True, tokens=257)
