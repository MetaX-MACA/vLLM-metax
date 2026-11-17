# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm.plugins import load_general_plugins

load_general_plugins()

from vllm.model_executor import parameter
from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import MPLinearLayerConfig
from vllm.model_executor.parameter import GroupQuantScaleParameter, PackedvLLMParameter
from vllm.scalar_type import scalar_types
from vllm_metax.registry.linear_kernels import mp


@pytest.fixture(scope="module")
def numerical_results():
    results = []
    yield results
    destination = os.environ.get("BRANCH_PORT_NUMERIC_RESULTS")
    if destination:
        import vllm_metax._C_stable_libtorch as extension
        Path(destination).write_text(json.dumps({
            "cases": results,
            "extension": extension.__file__,
            "extension_sha256": hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
            "gemm_schema": str(torch.ops._C.gptq_gemm.default._schema),
            "shuffle_schema": str(torch.ops._C.gptq_shuffle.default._schema),
            "provider": "source", "use_precompiled_kernel": os.environ.get("USE_PRECOMPILED_KERNEL"),
            "loaded_mcoplib_modules": [name for name in __import__("sys").modules if name == "mcoplib" or name.startswith("mcoplib.")],
        }, indent=2) + "\n")


def _pack_rows(values, bits):
    factor = 32 // bits
    # Independent integer shifts; no production pack/shuffle code in the reference.
    words = torch.zeros(values.shape[0] // factor, values.shape[1], dtype=torch.int64)
    for lane in range(factor):
        words |= values[lane::factor].to(torch.int64) << (bits * lane)
    return words.to(torch.int32)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("has_zero_points", [False, True])
def test_post_load_and_apply_match_independent_dequantization(monkeypatch, numerical_results, bits, group_size, dtype, has_zero_points):
    assert os.environ.get("USE_PRECOMPILED_KERNEL") == "0"
    # Model-free TP=1 parameter construction; all transformations and kernels are real.
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: 1)
    generator = torch.Generator().manual_seed(0)
    K, N = 256, 128
    groups = K // group_size
    codes = torch.randint(0, 1 << bits, (K, N), generator=generator)
    zeros = torch.full((groups, N), 1 << (bits - 1), dtype=torch.int64)
    scales = (torch.rand(groups, N, generator=generator) * 0.02 + 0.01).to(dtype)
    config = MPLinearLayerConfig(full_weight_shape=(K, N), partition_weight_shape=(K, N), weight_type=scalar_types.uint4b8 if bits == 4 else scalar_types.uint8b128, act_type=dtype, group_size=group_size, zero_points=has_zero_points)
    assert not hasattr(config, "has_g_idx")
    assert mp.MacaExllamaLinearKernel.can_implement(config) == (True, None)
    layer = torch.nn.Module()
    layer.register_parameter("qweight", PackedvLLMParameter(data=_pack_rows(codes, bits).cuda(), input_dim=0, output_dim=1, packed_dim=0, packed_factor=32 // bits, weight_loader=lambda *args: None))
    layer.register_parameter("scales", GroupQuantScaleParameter(data=scales.cuda(), input_dim=0, output_dim=1, weight_loader=lambda *args: None))
    if has_zero_points:
        layer.register_parameter("qzeros", torch.nn.Parameter(_pack_rows((zeros - 1).t(), bits).t().contiguous().cuda(), requires_grad=False))
    kernel = mp.MacaExllamaLinearKernel(config, "qweight", "scales", "qzeros" if has_zero_points else None)
    assert torch.empty(0, dtype=torch.int32, device=layer.qweight.device).data_ptr() == 0
    kernel.process_weights_after_loading(layer)
    assert len(kernel._get_weight_params(layer)) == 3
    group_ids = torch.arange(K) // group_size
    weight = (codes.double() - zeros[group_ids].double()) * scales[group_ids].double()
    for shape in [(1, K), (2, 3, K)]:
        x_cpu = (torch.randn(shape, generator=generator) * 0.1).to(dtype)
        bias_cpu = (torch.randn(N, generator=generator) * 0.03).to(dtype) if has_zero_points else None
        reference = x_cpu.double() @ weight
        if bias_cpu is not None:
            reference += bias_cpu.double()
        actual = kernel.apply_weights(layer, x_cpu.cuda(), None if bias_cpu is None else bias_cpu.cuda()).cpu()
        assert torch.isfinite(actual).all()
        atol = rtol = 0.02 if dtype == torch.bfloat16 else 0.005
        torch.testing.assert_close(actual.double(), reference, atol=atol, rtol=rtol)
        maximum = (actual.double() - reference).abs().max().item()
        numerical_results.append({"id": f"bits{bits}-group{group_size}-{dtype}-zp{has_zero_points}-shape{shape[:-1]}", "max_abs": maximum, "reference_scale": reference.abs().max().item(), "atol": atol, "rtol": rtol, "nan_count": 0, "inf_count": 0, "passed": True})


@pytest.mark.parametrize("names", [["q_weight", "bit"], ["q_weight", "q_perm", "bit"]])
def test_shuffle_forwards_only_known_schemas(monkeypatch, names):
    calls = []
    class Shuffle:
        _schema = SimpleNamespace(arguments=[SimpleNamespace(name=name) for name in names])
        def __call__(self, *args):
            calls.append(args)
    monkeypatch.setattr(torch.ops._C, "gptq_shuffle", SimpleNamespace(default=Shuffle()))
    weight = torch.empty(1)
    mp._gptq_shuffle(weight, 4)
    assert calls[0][0] is weight and calls[0][-1] == 4
    if len(names) == 3:
        assert calls[0][1].device == weight.device and calls[0][1].numel() == 0


def test_shuffle_does_not_hide_execution_errors(monkeypatch):
    class Shuffle:
        _schema = SimpleNamespace(arguments=[SimpleNamespace(name="q_weight"), SimpleNamespace(name="bit")])
        def __call__(self, *args):
            raise RuntimeError("native execution failed")
    monkeypatch.setattr(torch.ops._C, "gptq_shuffle", SimpleNamespace(default=Shuffle()))
    with pytest.raises(RuntimeError, match="native execution failed"):
        mp._gptq_shuffle(torch.empty(1), 4)


def test_shuffle_rejects_unknown_schema(monkeypatch):
    operation = SimpleNamespace(_schema=SimpleNamespace(arguments=[SimpleNamespace(name="unknown")]))
    monkeypatch.setattr(torch.ops._C, "gptq_shuffle", SimpleNamespace(default=operation))
    with pytest.raises(RuntimeError, match="Unsupported GPTQ shuffle schema"):
        mp._gptq_shuffle(torch.empty(1), 4)


@pytest.mark.parametrize("weight_type", [scalar_types.uint4, scalar_types.uint8])
def test_asymmetric_unbiased_types_are_not_selected(weight_type):
    config = MPLinearLayerConfig(full_weight_shape=(256, 128), partition_weight_shape=(256, 128), weight_type=weight_type, act_type=torch.float16, group_size=128, zero_points=True)
    supported, reason = mp.MacaExllamaLinearKernel.can_implement(config)
    assert not supported and "Quant type" in reason
