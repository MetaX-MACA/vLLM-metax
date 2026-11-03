# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# 用途: 在 MACA 上让 MoE-LoRA 的 one-shot Triton kernel 走 2-stage 流水线, 避免
#     共享内存超限(LoRA on MoE experts, 如 Qwen3-MoE + --enable-lora)。
#
# Note: `_run_fused_moe_lora_one_shot` lowers `num_stages` from 3 to 2 when the
#     device reports less than 68KB of shared memory per block. That branch does
#     run on MACA (`MacaPlatformBase.is_cuda_alike()` returns True), but the
#     reported limit does not match the limit Triton enforces:
#     `get_max_shared_memory_bytes` returns the driver opt-in attribute
#     (98304 bytes) while the kernel launch is validated against 65536 bytes, so
#     the kernel keeps num_stages=3 and fails with
#     `triton.runtime.errors.OutOfResources` during profile_run.
#
#     The v0.29.0 line additionally replaced the module's `current_platform`
#     with a stand-in whose `is_cuda_alike()` returns True. That half is
#     redundant here (the MACA platform already returns True) and would be
#     unsafe on v0.30.0: the patch installer repoints same-name exports in every
#     already-loaded module whose name starts with `vllm` (including MetaX) by
#     object identity, so a stand-in object
#     would spread to unrelated `current_platform` callers.
#
# Affected versions: vLLM 0.30.0 (ced6857afa0e), ported 2026-10-15 from
#     mx/v0.29.0-dev (98dff2a81d, verified 2026-09-29).
#
# Remove at: Upstream derives the pipeline depth from a shared memory limit that
#     Triton also honours on MACA.
# -----------------------------------------------------------------------------
"""Apply the MoE-LoRA one-shot kernel's 2-stage shared-memory fallback on MACA.
Affected: vLLM 0.30.0 (ced6857afa0e). `_run_fused_moe_lora_one_shot` in
`vllm/lora/ops/triton_ops/fused_moe_lora_op.py` lowers `num_stages` to 2 only
when `get_max_shared_memory_bytes(device.index)` reports less than 68KB. The
helper reads `_custom_ops.get_max_shared_memory_per_block_device_attribute`,
which reports the driver's opt-in size (98304 bytes) on MACA while Triton
validates the launch against the real 65536-byte per-block limit, so the
upstream fallback never triggers.
The module is patched at its own attribute level, so the LoRA MoE op is the
intended consumer. The installer may also repoint other loaded modules with names
starting with `vllm` (including `vllm_metax`) that imported the same helper, which
only lowers the reported limit to the value Triton enforces for the device.
"""

from __future__ import annotations
import torch
from vllm_metax.patch.utils import patch

_TARGET_MODULE = "vllm.lora.ops.triton_ops.fused_moe_lora_op"
# Conservative fallback when device properties cannot be queried.
_MACA_MAX_SHARED_MEMORY_BYTES = 64 * 1024


@patch(_TARGET_MODULE, "get_max_shared_memory_bytes")
def _get_max_shared_memory_bytes(gpu: int = 0) -> int:
    """Regular shared memory per block reported for `gpu`."""
    try:
        props = torch.cuda.get_device_properties(gpu)
    except Exception:  # noqa: BLE001 - properties may be unavailable on MACA
        return _MACA_MAX_SHARED_MEMORY_BYTES
    limit = getattr(props, "shared_memory_per_block", None)
    if limit:
        return int(limit)
    return _MACA_MAX_SHARED_MEMORY_BYTES
