# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Select MetaX FlashAttention 2 without probing NVIDIA implementations, while
#     accepting the current upstream version-selection arguments.
#
# Affected versions: vLLM 0.29.1.dev0 (98dff2a81d), verified 2026-09-17.
#
# Remove at: Upstream FlashAttention version selection dispatches through the MetaX
#     platform without probing NVIDIA implementations.
# -----------------------------------------------------------------------------

"""Select MetaX FlashAttention without probing NVIDIA implementations.

Affected: vLLM 0.29.1.dev0 (98dff2a81d). Remove when upstream dispatches FA
version selection through the platform. Keep its current keyword interface.
"""

from vllm_metax.v1.attention.backends.fa_utils import (
    get_flash_attn_version as _metax_get_flash_attn_version,
)
from vllm_metax.patch.utils import patch


@patch("vllm.v1.attention.backends.fa_utils")
def get_flash_attn_version(
    requires_alibi: bool = False,
    head_size: int | None = None,
    head_size_v: int | None = None,
    has_sinks: bool = False,
    requires_softcap: bool = False,
    kv_cache_block_size: int | None = None,
    supports_fa4_hd256: bool = False,
) -> int | None:
    return _metax_get_flash_attn_version(
        requires_alibi=requires_alibi,
        head_size=head_size,
        head_size_v=head_size_v,
        has_sinks=has_sinks,
    )
