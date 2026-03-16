# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# ------------------------------------------------------------------------
# Note: This file contains non-functional code changes (chores) for vLLM
#       to support the Metax platform.
#
# Remove the wrong error log for Maca when checking the flash attention version.
# ------------------------------------------------------------------------

import vllm
from vllm.v1.attention.backends.fa_utils import logger


def get_flash_attn_version(requires_alibi: bool = False) -> int | None:
    """
    Omit the wrong error log for Maca
    """
    logger.info_once(
        "Using Maca version of flash attention, which only supports version 2."
    )
    return 2


import vllm.v1.attention.backends.fa_utils

vllm.v1.attention.backends.fa_utils.get_flash_attn_version = get_flash_attn_version
