# SPDX-License-Identifier: Apache-2.0
import vllm
from vllm.logger import init_logger

logger = init_logger(__name__)

import vllm.device_allocator.cumem

from vllm_metax.device_allocator.cumem import (CuMemAllocator as
                                               mx_CuMemAllocator)

vllm.device_allocator.cumem.CuMemAllocator = mx_CuMemAllocator
