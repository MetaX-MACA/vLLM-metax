# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# 用途: 在 MACA 上让 MoE-LoRA 的 one-shot Triton kernel 走 2-stage 流水线, 避免
#     共享内存超限(LoRA on MoE experts, 如 Qwen3-MoE + --enable-lora)。
#
# Note: Upstream only shrinks num_stages inside
#     `if current_platform.is_cuda_alike():`, so on MACA the kernel keeps
#     num_stages=3 and requests 98304 bytes of shared memory while the device
#     limit is 65536 -> triton.runtime.errors.OutOfResources during profile_run.
#
# Affected versions: vLLM 0.29.1.dev0 (98dff2a81d), verified 2026-09-29.
#
# Remove at: Upstream derives the pipeline depth from the device shared memory
#     limit without gating it on `is_cuda_alike()`.
# -----------------------------------------------------------------------------

"""Apply the MoE-LoRA one-shot kernel's 2-stage shared-memory fallback on MACA.

Affected: vLLM 0.29.1.dev0 (98dff2a81d). `_run_fused_moe_lora_one_shot` in
`vllm/lora/ops/triton_ops/fused_moe_lora_op.py` computes
`num_stages` from the kernel heuristic and then only lowers it to 2 when
`current_platform.is_cuda_alike()` is true and the device reports less than 68KB
of shared memory per block. MACA is not CUDA-like, so the fallback never runs and
the kernel asks for 98304 bytes on a 65536-byte device.

The module is patched at its own attribute level, so only the LoRA MoE op is
affected and no other consumer of `current_platform` changes behaviour.
"""

from __future__ import annotations

import torch
from vllm.platforms import current_platform as _vllm_current_platform

from vllm_metax.patch.utils import patch

_TARGET_MODULE = "vllm.lora.ops.triton_ops.fused_moe_lora_op"

# MACA exposes 65536 bytes of shared memory per block (the value triton validates
# against); a larger "opt-in" attribute must not lift the 2-stage fallback.
_MACA_MAX_SHARED_MEMORY_BYTES = 64 * 1024


class _DelegatingPlatformMeta(type):
    """Forward attributes this proxy does not override to the real platform."""

    def __getattr__(cls, name):
        return getattr(_vllm_current_platform, name)


@patch(_TARGET_MODULE, "current_platform")
class _CudaAlikePlatform(metaclass=_DelegatingPlatformMeta):
    """`current_platform` stand-in whose CUDA-like gate is true.

    The LoRA MoE op module reads `is_cuda_alike()` (the shared-memory gate) plus
    `num_compute_units()`; the latter and any future attribute are delegated
    through the metaclass to the real platform instance.
    """

    @classmethod
    def is_cuda_alike(cls) -> bool:
        return True


@patch(_TARGET_MODULE, "get_max_shared_memory_bytes")
def _get_max_shared_memory_bytes(device: int = 0) -> int:
    """Shared memory per block of `device`, capped at MACA's real limit."""

    try:
        props = torch.cuda.get_device_properties(device)
    except Exception:  # noqa: BLE001 - properties may be unavailable on MACA
        return _MACA_MAX_SHARED_MEMORY_BYTES

    for attribute in ("shared_memory_per_block_optin", "shared_memory_per_block"):
        limit = getattr(props, attribute, None)
        if limit:
            return min(int(limit), _MACA_MAX_SHARED_MEMORY_BYTES)
    return _MACA_MAX_SHARED_MEMORY_BYTES
