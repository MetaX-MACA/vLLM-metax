# SPDX-License-Identifier: Apache-2.0

import vllm
from vllm_metax.patch.hook_registry import register_patch
from vllm.logger import init_logger

logger = init_logger(__name__)


MULTI_STEP_ATTENTION_BACKENDS = [
# ┌------------------------  Metax Modification -------------------------┐
    "FLASH_ATTN", "ROCM_FLASH", "FLASHINFER", "NO_ATTENTION", "TRITON_MLA"
# └------------------------- Metax Modification -------------------------┘
]

from vllm.worker import multi_step_model_runner

multi_step_model_runner.MULTI_STEP_ATTENTION_BACKENDS = MULTI_STEP_ATTENTION_BACKENDS