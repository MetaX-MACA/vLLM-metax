# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import ctypes

import pytest

from vllm_metax.distributed.device_communicators import all2all
from vllm_metax.distributed.device_communicators.cuda_communicator import MacaCommunicator
from vllm_metax.distributed.device_communicators.cuda_wrapper import CudaRTLibrary
from vllm_metax.distributed.device_communicators.pynccl_wrapper import MCCLLibrary


def test_cuda_rt_library_maps_cuda_calls_to_maca_symbols():
    mapping = CudaRTLibrary.cuda_to_maca_mapping
    assert mapping["cudaMalloc"] == "mcMalloc"
    assert mapping["cudaMemcpy"] == "mcMemcpy"
    assert mapping["cudaIpcGetMemHandle"] == "mcIpcGetMemHandle"
    assert mapping["cudaHostRegister"] == "mcHostRegister"
    assert mapping["mcExtMallocWithFlags"] == "mcExtMallocWithFlags"


@pytest.mark.requires_maca
def test_cuda_rt_library_allocates_and_frees_maca_memory():
    library = CudaRTLibrary()
    ptr = library.cudaMalloc(4096)
    try:
        assert ptr.value
        library.cudaMemset(ptr, 0, 4096)
        library.cudaDeviceSynchronize()
    finally:
        library.cudaFree(ptr)


def test_maca_communicator_uses_plugin_class():
    assert MacaCommunicator.__module__ == "vllm_metax.distributed.device_communicators.cuda_communicator"


def test_maca_all2all_managers_override_upstream_dispatch():
    assert all2all.MacaAgRsAll2AllManager.__module__.startswith("vllm_metax.")
    assert all2all.MacaDeepEPHTAll2AllManager.__module__.startswith("vllm_metax.")
    assert all2all.MacaDeepEPLLAll2AllManager.__module__.startswith("vllm_metax.")


def test_mccl_library_uses_maca_symbol_namespace():
    exported_names = {function.name for function in MCCLLibrary.exported_functions}
    assert exported_names
    assert all(name.startswith("mccl") for name in exported_names)
    assert {"mcclGetVersion", "mcclAllReduce", "mcclAllGather"} <= exported_names
    assert not {name for name in exported_names if name.startswith("nccl")}
