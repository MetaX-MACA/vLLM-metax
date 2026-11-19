# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch

import vllm_metax.device_allocator.cumem as plugin_cumem
from vllm_metax.device_allocator import get_mem_allocator_instance
from vllm_metax.device_allocator.sleep_mode_backend import MacaMemBackend


def test_cumem_module_is_replaced_in_upstream_namespace():
    import vllm.device_allocator.cumem as upstream_cumem

    assert upstream_cumem is plugin_cumem
    assert upstream_cumem.CuMemAllocator is plugin_cumem.CuMemAllocator


def test_platform_allocator_factory_returns_plugin_allocator():
    assert get_mem_allocator_instance() is plugin_cumem.CuMemAllocator.get_instance()


def test_macamem_backend_registration_and_semantics():
    from vllm.device_allocator.sleep_mode_backend import SleepModeBackendFactory

    assert SleepModeBackendFactory.get_backend_class("macamem") is MacaMemBackend
    assert MacaMemBackend.preserves_communicators() is True


@pytest.mark.requires_maca
def test_plugin_allocator_offloads_and_restores_weight_tag():
    allocator = plugin_cumem.CuMemAllocator.get_instance()
    expected = torch.arange(1024, device="cuda", dtype=torch.float32)
    try:
        with allocator.use_memory_pool(tag="weights"):
            actual = expected.clone()
        allocator.sleep(offload_tags=("weights",))
        allocator.wake_up(("weights",))
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected)
    finally:
        allocator.close()
