# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek-V4 FlashMLA sparse backend, metadata, and metadata builder."""

from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.platforms.interface import DeviceCapability

from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLABackend


class MacaDeepseekV4FlashMLABackend(DeepseekV4FlashMLABackend):
    """DeepSeek-V4 sparse-MLA backend.

    Subclasses ``AttentionBackend`` directly (not the V3.2
    ``FlashMLASparseBackend``): DeepSeek-V4 runs its own attention layer
    (``DeepseekV4Attention``), so it does not reuse the V3.2 builder or impl, and
    only needs to declare its own metadata builder, KV-cache layout, and the
    sparse-MLA capability flags.
    """

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        # "fp8_ds_mla",
        # "fp8",  # alias for fp8_ds_mla
    ]

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return True
