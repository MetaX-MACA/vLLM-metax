# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Optional
from vllm.attention.utils.fa_utils import logger

import vllm.attention.utils.fa_utils


def get_flash_attn_version(requires_alibi: bool = False) -> Optional[int]:
    """
    vllm_metax directly uses flash_attn via `import flash_attn` instead of vLLM's vll_flash_attn module,
    making the get_flash_attn_version function obsolete.
    When eager.py imports FlashAttentionMetadata, the import phase triggers get_flash_attn_version,
    which causes incorrect log output.
    """
    logger.info_once(
        "Using Maca version of flash attention, which only supports version 2."
    )
    return 2


vllm.attention.utils.fa_utils.get_flash_attn_version = get_flash_attn_version
