# 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import vllm
from vllm_metax_plugin.patch.hook_registry import register_module_redirect
from vllm.logger import init_logger

logger = init_logger(__name__)

import vllm_metax_plugin.device_allocator.cumem
import vllm.device_allocator

vllm.device_allocator.cumem = vllm_metax_plugin.device_allocator.cumem

register_module_redirect(
    "vllm.device_allocator.cumem",
    "vllm_metax_plugin.device_allocator.cumem")