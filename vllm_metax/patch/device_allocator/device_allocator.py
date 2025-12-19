# SPDX-License-Identifier: Apache-2.0

# ------------------------------------------------------------
# Note: this patch is to support sleep_mode on maca backend.
#       We need to replace cuda `CuMemAllocator` with maca's.
# ------------------------------------------------------------


import vllm
from vllm.logger import init_logger

logger = init_logger(__name__)

from contextlib import AbstractContextManager, nullcontext
from vllm.utils.mem_constants import GiB_bytes

import torch

# from vllm.v1.worker import worker_base
from vllm.v1.worker.gpu_worker import Worker
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.distributed.kv_transfer import ensure_kv_transfer_initialized


def sleep(self, level: int = 1) -> None:
    from vllm_metax.device_allocator.cumem import CuMemAllocator

    free_bytes_before_sleep = torch.cuda.mem_get_info()[0]

    # Save the buffers before level 2 sleep
    if level == 2:
        model = self.model_runner.model
        self._sleep_saved_buffers = {
            name: buffer.cpu().clone() for name, buffer in model.named_buffers()
        }

    allocator = CuMemAllocator.get_instance()
    allocator.sleep(offload_tags=("weights",) if level == 1 else tuple())
    free_bytes_after_sleep, total = torch.cuda.mem_get_info()
    freed_bytes = free_bytes_after_sleep - free_bytes_before_sleep
    used_bytes = total - free_bytes_after_sleep
    assert freed_bytes >= 0, "Memory usage increased after sleeping."
    logger.info(
        "Sleep mode freed %.2f GiB memory, %.2f GiB memory is still in use.",
        freed_bytes / GiB_bytes,
        used_bytes / GiB_bytes,
    )


def wake_up(self, tags: list[str] | None = None) -> None:
    from vllm_metax.device_allocator.cumem import CuMemAllocator

    allocator = CuMemAllocator.get_instance()
    allocator.wake_up(tags)

    # Restore the buffers after level 2 sleep
    if len(self._sleep_saved_buffers):
        model = self.model_runner.model
        for name, buffer in model.named_buffers():
            if name in self._sleep_saved_buffers:
                buffer.data.copy_(self._sleep_saved_buffers[name].data)
        self._sleep_saved_buffers = {}

    # If the KV cache has just been woken up,
    # the internal state of cache_engine must be reset,
    # especially the FP8 scaling factor.
    if (
        (tags is None or "kv_cache" in tags)
        and self.cache_config.cache_dtype.startswith("fp8")
        and hasattr(self.model_runner, "init_fp8_kv_scales")
    ):
        self.model_runner.init_fp8_kv_scales()


def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager:
    if self.vllm_config.model_config.enable_sleep_mode:
        from vllm_metax.device_allocator.cumem import CuMemAllocator

        allocator = CuMemAllocator.get_instance()
        if tag == "weights":
            assert allocator.get_current_usage() == 0, (
                "Sleep mode can only be used for one instance per process."
            )
        return allocator.use_memory_pool(tag=tag)
    else:
        return nullcontext()


def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
    """Allocate GPU KV cache with the specified kv_cache_config."""

    # Init kv cache connector here, because it requires
    # `kv_cache_config`.
    # NOTE(Kuntai): This need to be done before `initialize_kv_cache`,
    # because `initialize_kv_cache` will inject kv cache groups not
    # related to kv cache connector (e.g. kv cache sharing layers).
    ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)

    if self.vllm_config.model_config.enable_sleep_mode:
        from vllm_metax.device_allocator.cumem import CuMemAllocator

        allocator = CuMemAllocator.get_instance()
        with allocator.use_memory_pool(tag="kv_cache"):
            self.model_runner.initialize_kv_cache(kv_cache_config)
    else:
        self.model_runner.initialize_kv_cache(kv_cache_config)


Worker.sleep = sleep
Worker.wake_up = wake_up
Worker._maybe_get_memory_pool_context = _maybe_get_memory_pool_context
Worker.initialize_from_config = initialize_from_config
