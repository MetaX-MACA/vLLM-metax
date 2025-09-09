# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

from vllm_metax import _custom_ops as ops

reshape_and_cache_flash = ops.reshape_and_cache_flash

get_scheduler_metadata = None


def get_flash_attn_version(requires_alibi: bool = False) -> Optional[int]:
    # import here to avoid circular dependencies
    return None


def flash_attn_supports_fp8() -> bool:
    return False


def is_flash_attn_varlen_func_available() -> bool:
    return True
