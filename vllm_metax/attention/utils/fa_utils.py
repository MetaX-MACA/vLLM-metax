# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional

from vllm import _custom_ops as ops
from vllm.attention.utils.fa_utils import logger
from vllm.platforms import current_platform

get_scheduler_metadata = None

if current_platform.is_out_of_tree():
    from vllm import _custom_ops as ops
    reshape_and_cache_flash = ops.reshape_and_cache_flash
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache  # noqa: F401
    get_scheduler_metadata = None


def get_flash_attn_version(requires_alibi: bool = False) -> Optional[int]:
    logger.info_once(
        "Using Maca version of flash attention, which only supports version 2."
    )
    return None


def flash_attn_supports_fp8() -> bool:
    logger.info_once(
        "Using Maca version of flash attention, which does not support FP8")
    return False


def flash_attn_supports_mla():
    return False


def is_flash_attn_varlen_func_available() -> bool:
    return True
