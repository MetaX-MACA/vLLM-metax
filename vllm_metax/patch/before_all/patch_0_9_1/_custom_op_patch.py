# SPDX-License-Identifier: Apache-2.0

import sys 

import vllm
from vllm.logger import init_logger
from vllm_metax import _custom_ops as _metax_custom_ops
from vllm_metax.patch.hook_registry import register_module_redirect

logger = init_logger(__name__)

vllm._custom_ops = _metax_custom_ops;

register_module_redirect("vllm._custom_ops", "vllm_metax._custom_ops")

