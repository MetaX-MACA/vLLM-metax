# SPDX-License-Identifier: Apache-2.0

from vllm.attention.utils.fa_utils import logger

import vllm.attention.utils.fa_utils


def get_flash_attn_version(requires_alibi: bool = False) -> int | None:
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
