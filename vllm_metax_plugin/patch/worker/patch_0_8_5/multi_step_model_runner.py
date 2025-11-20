# 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import vllm
from vllm_metax_plugin.patch.hook_registry import register_patch
from vllm.logger import init_logger

logger = init_logger(__name__)


MULTI_STEP_ATTENTION_BACKENDS = [
    "FLASH_ATTN", "ROCM_FLASH", "FLASHINFER", "NO_ATTENTION", "TRITON_MLA"
]

from vllm.worker import multi_step_model_runner

multi_step_model_runner.MULTI_STEP_ATTENTION_BACKENDS = MULTI_STEP_ATTENTION_BACKENDS