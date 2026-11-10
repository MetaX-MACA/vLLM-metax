# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashAttention version selection and unified MetaX exports."""

from __future__ import annotations

from vllm import _custom_ops as ops
from vllm.config import get_current_vllm_config_or_none
from vllm.config.cache import CacheDType
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.fa_utils import logger

from vllm_metax.third_party.fa_interface import (
    FA2_SUPPORTED_HDIM,  # noqa: F401
    FA3_SUPPORTED_HDIM_BF16,  # noqa: F401
    FA3_SUPPORTED_HDIM_FP8,  # noqa: F401
    fa_version_unsupported_reason,
    flash_attn_varlen_func,  # noqa: F401
    flash_attn_with_kvcache,  # noqa: F401
    is_fa_geometry_supported,
    is_fa_version_supported,
)

from vllm_metax.utils.math_util import is_power_of_2

reshape_and_cache_flash = ops.reshape_and_cache_flash
compile_flash_attn_varlen_func_from_specs = None


def get_flash_attn_version(
    requires_alibi: bool = False,
    head_size: int | None = None,
    head_size_v: int | None = None,
    has_sinks: bool = False,
    kv_cache_block_size: int | None = None,
    *,
    kv_cache_dtype: CacheDType | None = None,
    device_capability: DeviceCapability | None = None,
) -> int | None:
    """Temporarily prefer FA2 and enable FA3 only on capability 8.9 devices."""
    if kv_cache_dtype not in (None, "auto", "float16", "bfloat16", "fp8", "fp8_e4m3"):
        return None
    capability = (
        device_capability
        if device_capability is not None
        else current_platform.get_device_capability()
    )
    if capability is None or capability < DeviceCapability(8, 0):
        return None
    is_fp8 = kv_cache_dtype in ("fp8", "fp8_e4m3")
    if is_fp8 and capability != DeviceCapability(8, 9):
        return None
    if kv_cache_block_size is not None and not is_power_of_2(kv_cache_block_size):
        return None

    config = get_current_vllm_config_or_none()
    fa_version = (
        config.attention_config.flash_attn_version if config is not None else None
    )
    if fa_version is None:
        # NOTE(metax): Temporary policy: prefer FA2; allow FA3 for FP8 or
        # when FA2 is unavailable, and only on FP8-capable MACA devices.
        fa_version = (
            3
            if capability == DeviceCapability(8, 9)
            and (is_fp8 or not is_fa_version_supported(2))
            else 2
        )
    if fa_version not in (2, 3):
        return None
    head_size_v = head_size if head_size_v is None else head_size_v

    reasons: list[str] = []
    if fa_version == 3:
        if capability != DeviceCapability(8, 9):
            reasons.append(
                "FA3 is temporarily enabled only on FP8-capable devices "
                f"(capability 8.9); got {capability}"
            )
        if not is_fa_version_supported(3):
            reasons.append(f"FA3 is unavailable: {fa_version_unsupported_reason(3)}")
        elif not is_fa_geometry_supported(3, head_size, head_size_v, is_fp8=is_fp8):
            reasons.append(
                f"unsupported QK/V head dimensions {head_size}/{head_size_v} "
                f"for kv_cache_dtype={kv_cache_dtype or 'auto'} "
                f"({'FP8' if is_fp8 else 'non-FP8'})"
            )
        if requires_alibi:
            reasons.append("ALiBi is not supported")
        if has_sinks:
            reasons.append("attention sinks are not supported")
        if reasons:
            fa_version = 2

    if fa_version == 2:
        if not is_fa_version_supported(2) or not is_fa_geometry_supported(
            2, head_size, head_size_v, is_fp8=is_fp8
        ):
            return None
        if reasons:
            logger.warning_once(
                "Cannot use MetaX FA3; falling back to FA2: %s", "; ".join(reasons)
            )
    logger.info_once("Using MetaX FlashAttention version %d", fa_version)
    return fa_version


def flash_attn_supports_kv_cache_dtype(
    kv_cache_dtype: CacheDType = "fp8_e4m3",
    *,
    requires_alibi: bool = False,
    head_size: int | None = None,
    head_size_v: int | None = None,
    has_sinks: bool = False,
    kv_cache_block_size: int | None = None,
) -> bool:
    return (
        get_flash_attn_version(
            requires_alibi=requires_alibi,
            head_size=head_size,
            head_size_v=head_size_v,
            has_sinks=has_sinks,
            kv_cache_block_size=kv_cache_block_size,
            kv_cache_dtype=kv_cache_dtype,
        )
        is not None
    )


def flash_attn_supports_quant_query_input(fa_version: int | None = None) -> bool:
    return fa_version == 3 and is_fa_version_supported(3)


def flash_attn_supports_sinks(fa_version: int | None = None) -> bool:
    return fa_version in (None, 2) and is_fa_version_supported(2)


def flash_attn_supports_mla() -> bool:
    return False


def is_flash_attn_varlen_func_available() -> bool:
    return is_fa_version_supported(2) or is_fa_version_supported(3)


def get_scheduler_metadata(*args, fa_version: int = 2, **kwargs):
    if fa_version == 2:
        return None
    raise NotImplementedError(
        f"MetaX FA{fa_version} scheduler metadata is not implemented"
    )


def uses_fa4_hd256_kernel(
    head_size: int | None, head_size_v: int | None = None
) -> bool:
    return False
