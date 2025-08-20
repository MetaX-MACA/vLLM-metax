import vllm
from vllm_metax.patch.hook_registry import register_module_redirect
from vllm.logger import init_logger

logger = init_logger(__name__)

import vllm_metax.device_allocator.cumem
import vllm.device_allocator

vllm.device_allocator.cumem = vllm_metax.device_allocator.cumem

register_module_redirect(
    "vllm.device_allocator.cumem",
    "vllm_metax.device_allocator.cumem")