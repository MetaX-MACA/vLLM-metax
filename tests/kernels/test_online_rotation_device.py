# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading

import pytest
import torch

from vllm_metax.model_executor.kernels.online_rotate import tilelang_kernels as rotation


def sylvester(size):
    matrix = torch.ones((1, 1), dtype=torch.float64)
    while matrix.shape[0] < size:
        matrix = torch.cat((torch.cat((matrix, matrix), 1),
                            torch.cat((matrix, -matrix), 1)), 0)
    return matrix


@pytest.fixture(scope="module")
def evidence():
    result = {"cases": [], "device_bindings": []}
    yield result
    result.update({"source_sha256": hashlib.sha256(Path(rotation.__file__).read_bytes()).hexdigest(),
                   "cache_keys": [list(key) for key in rotation._KERNEL_CACHE],
                   "provider": "source", "use_precompiled_kernel": os.environ.get("USE_PRECOMPILED_KERNEL"),
                   "loaded_mcoplib_modules": [n for n in sys.modules if n == "mcoplib" or n.startswith("mcoplib.")]})
    if destination := os.environ.get("BRANCH_PORT_NUMERIC_RESULTS"):
        Path(destination).write_text(json.dumps(result, indent=2) + "\n")


def tensors(gpu, dtype):
    generator = torch.Generator().manual_seed(0)
    x = torch.randn((3, 64), generator=generator, dtype=torch.float32).to(dtype)
    sign = (2 * torch.randint(0, 2, (64,), generator=generator) - 1).to(torch.int8)
    return (x.to(f"cuda:{gpu}"), sign.to(f"cuda:{gpu}"),
            torch.ones((1, 1), dtype=torch.float32, device=f"cuda:{gpu}"),
            sylvester(64).to(device=f"cuda:{gpu}", dtype=dtype))


def compare(x, sign, output, inverse, evidence, label):
    x_cpu, sign_cpu = x.cpu().double(), sign.cpu().double()
    h = sylvester(x.shape[-1]) / math.sqrt(x.shape[-1])
    expected = (x_cpu @ h) * sign_cpu if inverse else (x_cpu * sign_cpu) @ h
    actual = output.cpu().double()
    atol, rtol = (0.002, 0.005) if x.dtype == torch.float16 else (0.02, 0.02)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    evidence["cases"].append({"id": label, "gpu": x.device.index,
                              "dtype": str(x.dtype), "inverse": inverse,
                              "max_abs": (actual-expected).abs().max().item(),
                              "atol": atol, "rtol": rtol, "passed": True})


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("inverse", [False, True])
def test_compile_threads_and_per_device_cache(dtype, inverse, evidence, monkeypatch):
    assert torch.cuda.device_count() >= 2
    original_set_device = torch.cuda.set_device

    def trace_set_device(device):
        original_set_device(device)
        evidence["device_bindings"].append({"thread": threading.current_thread().name,
                                            "requested": str(device),
                                            "actual": torch.cuda.current_device()})

    monkeypatch.setattr(torch.cuda, "set_device", trace_set_device)
    inputs = {gpu: tensors(gpu, dtype) for gpu in (0, 1)}

    def invoke(gpu):
        # The caller intentionally starts on the other GPU.
        torch.cuda.set_device(1-gpu)
        x, sign, hk, h_gemm = inputs[gpu]
        output = rotation.hadamard(x, sign, hk, h_gemm, inverse=inverse, k=1, p=64)
        torch.cuda.synchronize(gpu)
        assert torch.cuda.current_device() == 1-gpu
        return output

    with ThreadPoolExecutor(max_workers=2, thread_name_prefix="rotation-caller") as pool:
        futures = {gpu: pool.submit(invoke, gpu) for gpu in (0, 1)}
        outputs = {gpu: future.result() for gpu, future in futures.items()}
    for gpu in (0, 1):
        compare(inputs[gpu][0], inputs[gpu][1], outputs[gpu], inverse, evidence,
                f"compile-{dtype}-{inverse}-{gpu}")
    assert (0, 64, 1, inverse, str(dtype).split(".")[-1], 64) in rotation._KERNEL_CACHE
    assert (1, 64, 1, inverse, str(dtype).split(".")[-1], 64) in rotation._KERNEL_CACHE
    before = dict(rotation._KERNEL_CACHE)
    compare(inputs[0][0], inputs[0][1], invoke(0), inverse, evidence,
            f"reentry-{dtype}-{inverse}-0")
    assert rotation._KERNEL_CACHE == before
    assert any(record["thread"].startswith("tilelang") or
               not record["thread"].startswith(("MainThread", "rotation-caller"))
               for record in evidence["device_bindings"])


@pytest.mark.parametrize("gpu", [0, 1])
def test_warm_graph_replay_uses_tensor_device(gpu, evidence):
    x, sign, hk, h_gemm = tensors(gpu, torch.float16)
    with torch.cuda.device(gpu):
        stream = torch.cuda.Stream(device=gpu)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                rotation.hadamard(x, sign, hk, h_gemm, inverse=False, k=1, p=64)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = rotation.hadamard(x, sign, hk, h_gemm, inverse=False, k=1, p=64)
        graph.replay()
        torch.cuda.synchronize(gpu)
        compare(x, sign, output, False, evidence, f"graph-first-{gpu}")
        x.mul_(0.5)
        graph.replay()
        torch.cuda.synchronize(gpu)
        compare(x, sign, output, False, evidence, f"graph-changed-{gpu}")
