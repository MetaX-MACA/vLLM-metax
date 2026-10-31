# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Shape-specific autotune configuration for the TileLang Hadamard kernel."""

from __future__ import annotations

from typing import Any

from tilelang.engine.param import KernelParam
from tilelang.utils.tensor import TensorSupplyType, get_tensor_supply


AUTOTUNE_TOKENS = 256


def get_hadamard_configs(
    hidden_size: int,
    base_order: int,
    inverse: bool,
    dtype: str = "bfloat16",
    num_threads: int = 256,
    gemm_order: int = 64,
    mma_rows: int = 0,
    gemm_policy: str = "square",
) -> list[dict[str, int | str]]:
    """Build ``threads x neighboring M tiles x policies`` candidates."""
    del base_order, inverse, dtype, num_threads, mma_rows, gemm_policy

    rows_per_token = hidden_size // gemm_order
    floor_rows = (rows_per_token // 32) * 32
    ceil_rows = ((rows_per_token + 31) // 32) * 32
    row_candidates = []
    for rows in (floor_rows, ceil_rows):
        if rows > 0 and rows not in row_candidates:
            row_candidates.append(rows)

    return [
        {
            "num_threads": threads,
            "mma_rows": rows,
            "gemm_policy": policy,
        }
        for threads in (512, 256)
        for rows in row_candidates
        for policy in ("square", "full_col")
    ]


def supply_hadamard_inputs(params: list[KernelParam]) -> list[Any]:
    """Create profiling inputs with a fixed dynamic token dimension."""
    supply = get_tensor_supply(TensorSupplyType.Auto)
    inputs = []
    for param in params:
        if param.is_scalar():
            inputs.append(AUTOTUNE_TOKENS)
            continue
        shape = [
            dim if isinstance(dim, int) else AUTOTUNE_TOKENS for dim in param.shape
        ]
        inputs.append(supply(KernelParam(dtype=param.dtype, shape=shape)))
    return inputs


__all__ = [
    "AUTOTUNE_TOKENS",
    "get_hadamard_configs",
    "supply_hadamard_inputs",
]
