# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Early compatibility hooks for vLLM and MetaX runtime mismatches."""

import torch
from torch.library import Library

_FRAGMENT_LIBS: list[Library] = []


def _has_op_overload(name: str, overload_name: str | None = None) -> bool:
    if not hasattr(torch.ops, "_C") or not hasattr(torch.ops._C, name):
        return False
    if overload_name is None:
        return True
    return hasattr(getattr(torch.ops._C, name), overload_name)


def _define_fragment(schema: str) -> None:
    try:
        lib = Library("_C", "FRAGMENT")
        lib.define(schema)
        _FRAGMENT_LIBS.append(lib)
    except Exception:
        # Another package or prior import may have registered it already.
        pass


if not _has_op_overload("scaled_fp4_quant", "out"):
    _define_fragment(
        "scaled_fp4_quant.out("
        "Tensor input, Tensor input_scale, bool is_sf_swizzled_layout, "
        "*, Tensor! output, Tensor! output_scale) -> ()"
    )

if not _has_op_overload("silu_and_mul_per_block_quant"):
    _define_fragment(
        "silu_and_mul_per_block_quant("
        "Tensor! output, Tensor input, Tensor! scales, int group_size, "
        "Tensor? scale_ub, bool is_scale_transposed) -> ()"
    )

if hasattr(torch, "accelerator"):
    cuda_module = getattr(torch, "cuda", None)
    if cuda_module is not None:
        for _name in (
            "current_device",
            "device_count",
            "empty_cache",
            "is_available",
            "max_memory_allocated",
            "mem_get_info",
            "memory_allocated",
            "memory_reserved",
            "memory_stats",
            "reset_peak_memory_stats",
            "set_device",
            "synchronize",
        ):
            if not hasattr(torch.accelerator, _name) and hasattr(cuda_module, _name):
                setattr(torch.accelerator, _name, getattr(cuda_module, _name))
        if (
            not hasattr(torch.accelerator, "current_device_index")
            and hasattr(cuda_module, "current_device")
        ):
            torch.accelerator.current_device_index = cuda_module.current_device
