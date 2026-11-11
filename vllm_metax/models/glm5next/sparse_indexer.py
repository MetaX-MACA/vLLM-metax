# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse attention indexer (kpool) — hardware-isolated entry point.

The implementation lives under ``nvidia/`` and ``amd/`` (with shared helpers
in ``common/``); this module picks the right one for the current platform.
"""

from vllm.platforms import current_platform


from vllm_metax.models.glm5next.metax.sparse_indexer import (
    MacaSparseAttnIndexerKpool,
)

__all__ = [
    "MacaSparseAttnIndexerKpool",
]
