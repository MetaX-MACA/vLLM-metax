# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

from functools import wraps, cache
from typing_extensions import ParamSpec
from typing import Callable, TypeVar
import threading

from tools.batched_test import pymxml as ml

_P = ParamSpec("_P")
_R = TypeVar("_R")

ml_available = False
try:
    try:
        ml.nvmlInit()
        ml_available = True
    except Exception:
        # Use NVIDIA's bindings when the MetaX management library is unavailable.
        import vllm.third_party.pynvml as ml

        ml.nvmlInit()
        ml_available = True
        print(f"[WARN] mx NVML is not available, but use nvidia NVML instead of it.")
finally:
    if ml_available:
        ml.nvmlShutdown()

assert ml_available, "mx NVML and nvidia NVML are both not available on this system."


def with_mxml_context(fn: Callable[_P, _R]) -> Callable[_P, _R]:
    @wraps(fn)
    def wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        ml.nvmlInit()
        try:
            return fn(*args, **kwargs)
        finally:
            ml.nvmlShutdown()

    return wrapper


class GPUManager:
    def __init__(self, max_idle_mem_mb=900):
        # Maximum used device memory in MiB for a GPU to count as idle.
        self.max_idle_mem = max_idle_mem_mb
        # Protects occupied_gpus and allocation/release decisions within this batch.
        self.global_mutex = threading.Lock()
        # Physical GPU indices reserved by this batch, regardless of current memory use.
        self.occupied_gpus: set[int] = set()

    @cache
    @with_mxml_context
    def get_gpu_count(self) -> int:
        return ml.nvmlDeviceGetCount()

    @with_mxml_context
    def get_gpu_memory_list(self) -> list[dict]:
        """
        Get a list of dict, length equal to the number of GPUs, each dict contains used, free, and total memory (MiB).
        Uses pynvml to get memory usage.
        """
        gpu_count = self.get_gpu_count()
        mems_infos = []
        for i in range(gpu_count):
            handle = ml.nvmlDeviceGetHandleByIndex(i)
            info = ml.nvmlDeviceGetMemoryInfo(handle)
            mems_infos.append(
                {
                    "used": int(info.used) // (1024 * 1024),  # type: ignore
                    "free": int(info.free) // (1024 * 1024),  # type: ignore
                    "total": int(info.total) // (1024 * 1024),  # type: ignore
                }
            )
        return mems_infos

    @with_mxml_context
    def get_free_gpu_indices(self, used_threshold_mb: int | None = None) -> list[int]:
        """
        Get unreserved GPUs whose used memory is at most the threshold (MiB).
        """
        if used_threshold_mb is None:
            used_threshold_mb = self.max_idle_mem

        mem_info = self.get_gpu_memory_list()
        free_list = [
            i
            for i, mem in enumerate(mem_info)
            if mem["used"] <= used_threshold_mb and i not in self.occupied_gpus
        ]
        return free_list

    def allocate(self, num_required: int) -> list[int]:
        """
        !!!LOCKING!!!
        Allocate GPUs based on current free memory status.
        Returns a list of allocated GPU indices.
        """
        if num_required > self.get_gpu_count():
            raise ValueError("Requested more GPUs than available on the system.")

        with self.global_mutex:
            free_gpus = self.get_free_gpu_indices()
            if len(free_gpus) < num_required:
                return []
            allocated = free_gpus[:num_required]
            self.occupied_gpus.update(allocated)
            return allocated

    def release(self, gpu_indices: list[int]) -> None:
        """
        !!!LOCKING!!!
        Release previously allocated GPUs.
        """
        with self.global_mutex:
            for idx in gpu_indices:
                self.occupied_gpus.discard(idx)
