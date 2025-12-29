# SPDX-License-Identifier: Apache-2.0
from typing import Callable
from vllm.attention.backends.registry import AttentionBackendEnum


def maybe_get_vit_flash_attn_backend(
    attn_backend: AttentionBackendEnum | None,
) -> Callable | None:
    # At this point,
    # we already have the attn_backend,
    # overriding logic is done in the platform-specific implementation.
    # so we don't need to override backend here.
    # Just return the attn_backend and flash_attn_varlen_func.

    if attn_backend == AttentionBackendEnum.FLASH_ATTN:
        from vllm_metax.attention.utils.fa_utils import flash_attn_varlen_func

    # if attn_backend is TORCH_SDPA,
    # it will reach here and the flash_attn_varlen_func will be None.
    return flash_attn_varlen_func


import vllm.attention.layer

vllm.attention.layer.maybe_get_vit_flash_attn_backend = maybe_get_vit_flash_attn_backend
