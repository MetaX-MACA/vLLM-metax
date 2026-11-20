# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch
import torch.multiprocessing as mp

from vllm_metax.device_allocator import get_mem_allocator_instance
from vllm_metax.device_allocator.sleep_mode_backend import MacaMemBackend


# Adapted from vLLM v0.33.0 tests/basic_correctness/memory/cumem/test_cumem.py.
@pytest.mark.requires_maca
@pytest.mark.parametrize("level", [1, 2])
def test_maca_sleep_selectively_restores_pools(device, level):
    allocator = get_mem_allocator_instance()
    backend = MacaMemBackend()
    reference = torch.arange(4096, device=device, dtype=torch.int32)
    weights = runtime = cache = None
    try:
        with allocator.use_memory_pool("weights"):
            weights = reference.clone()
        with allocator.use_memory_pool("runtime"):
            runtime = reference + 1
        with allocator.use_memory_pool("kv_cache"):
            cache = reference + 2

        allocations = {
            data.tag: data
            for data in allocator.pointer_to_data.values()
            if data.tag in {"weights", "runtime", "kv_cache"}
        }
        assert set(allocations) == {"weights", "runtime", "kv_cache"}

        backend.suspend(level)
        assert backend.state() == "SUSPENDED"
        assert all(data.is_asleep for data in allocations.values())
        assert (allocations["weights"].cpu_backup_tensor is not None) == (level == 1)
        assert allocations["runtime"].cpu_backup_tensor is not None
        assert allocations["kv_cache"].cpu_backup_tensor is None

        backend.resume(tags=["kv_cache"])
        assert backend.state() == "RUNNING"
        assert allocations["weights"].is_asleep
        assert not allocations["runtime"].is_asleep
        assert not allocations["kv_cache"].is_asleep
        torch.testing.assert_close(runtime, reference + 1, atol=0, rtol=0)
        cache.copy_(reference)
        torch.testing.assert_close(cache, reference, atol=0, rtol=0)

        backend.resume(tags=["weights"])
        assert all(not data.is_asleep for data in allocations.values())
        if level == 1:
            torch.testing.assert_close(weights, reference, atol=0, rtol=0)
        else:
            weights.copy_(reference)
            torch.testing.assert_close(weights, reference, atol=0, rtol=0)
    finally:
        backend.resume()
        del weights, runtime, cache
        allocator.close()


def _check_maca_sleep_graph(_rank, device):
    allocator = get_mem_allocator_instance()
    backend = MacaMemBackend()
    graph = torch.cuda.CUDAGraph()
    reference = torch.arange(4096, device=device, dtype=torch.int32)
    weights = cache = None
    try:
        with allocator.use_memory_pool("weights"):
            weights = reference.clone()
        with allocator.use_memory_pool("kv_cache"):
            cache = torch.empty_like(reference)

        graph_input = torch.ones_like(reference)
        with torch.cuda.graph(graph):
            torch.add(graph_input, weights, out=cache)
        torch.cuda.synchronize()
        addresses = weights.data_ptr(), cache.data_ptr()

        backend.suspend()
        backend.resume()
        assert (weights.data_ptr(), cache.data_ptr()) == addresses
        graph_input.fill_(41)
        graph.replay()
        torch.testing.assert_close(cache, reference + 41, atol=0, rtol=0)
    finally:
        graph.reset()
        backend.resume()
        del weights, cache
        allocator.close()


@pytest.mark.requires_maca
def test_maca_sleep_keeps_graph_addresses_replayable(device):
    # Keep capture streams and allocator state out of the pytest process.
    context = mp.spawn(_check_maca_sleep_graph, args=(device,), nprocs=1, join=False)
    process = context.processes[0]
    try:
        if not context.join(timeout=60):
            pytest.fail("allocator graph replay exceeded 60 seconds")
    finally:
        if process.is_alive():
            process.kill()
        process.join()
