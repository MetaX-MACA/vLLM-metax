# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# -----------------------------------------------
# Note: Replace CUDA `CuMemAllocator` with the MetaX allocator to support
#       sleep mode on MACA.
#
# Affected versions: v0.21.0
# -----------------------------------------------
import sys

from vllm.device_allocator import MemAllocator

# Redirect upstream cumem import to MACA version to avoid libcuda.so.1 dependency
import vllm_metax.device_allocator.cumem as _maca_cumem  # noqa: E402

sys.modules["vllm.device_allocator.cumem"] = _maca_cumem


def get_mem_allocator_instance() -> MemAllocator:
    from vllm_metax.device_allocator.cumem import CuMemAllocator

    return CuMemAllocator.get_instance()


import vllm.device_allocator

vllm.device_allocator.get_mem_allocator_instance = get_mem_allocator_instance
