# 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import sys 

import vllm
from vllm.logger import init_logger
from vllm_metax_plugin import _custom_ops as _metax_custom_ops
from vllm_metax_plugin.patch.hook_registry import register_module_redirect

logger = init_logger(__name__)

vllm._custom_ops = _metax_custom_ops;

register_module_redirect("vllm._custom_ops", "vllm_metax_plugin._custom_ops")

