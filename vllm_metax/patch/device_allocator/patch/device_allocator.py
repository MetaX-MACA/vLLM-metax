import vllm
from vllm.logger import init_logger

logger = init_logger(__name__)

from vllm_metax.device_allocator.cumem import CuMemAllocator as mx_CuMemAllocator
import vllm.device_allocator.cumem

vllm.device_allocator.cumem.CuMemAllocator = mx_CuMemAllocator
