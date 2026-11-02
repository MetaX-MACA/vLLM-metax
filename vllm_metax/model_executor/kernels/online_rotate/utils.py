# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Shape selection and matrix construction for Hadamard kernels."""

from __future__ import annotations


import torch


KNOWN_HADAMARD_ORDERS = (1, 12, 20, 28, 36, 40, 52, 60, 108, 140, 156, 172)
GEMM_HADAMARD_ORDERS = (16, 32, 64)


def is_pow2(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def factor_hadamard_dim(hidden_size: int) -> tuple[int, int]:
    if hidden_size <= 0:
        raise ValueError("hidden_size must be positive")
    for base_order in KNOWN_HADAMARD_ORDERS:
        if hidden_size % base_order == 0:
            width = hidden_size // base_order
            if is_pow2(width):
                return base_order, width
    raise ValueError(
        f"hidden_size={hidden_size} cannot be factored as K * 2^m with "
        f"supported K={KNOWN_HADAMARD_ORDERS}"
    )


def resolve_hadamard_dim(
    hidden_size: int,
    k: int | None = None,
    p: int | None = None,
) -> tuple[int, int]:
    if k is None and p is None:
        return factor_hadamard_dim(hidden_size)
    if k is None:
        if p is None or p <= 0 or hidden_size % p:
            raise ValueError("P must divide hidden_size")
        k = hidden_size // p
    if p is None:
        if k <= 0 or hidden_size % k:
            raise ValueError("K must divide hidden_size")
        p = hidden_size // k
    if k <= 0 or p <= 0 or k * p != hidden_size or not is_pow2(p):
        raise ValueError("Hadamard shape must satisfy hidden_size = K * 2^m")
    return k, p


def select_gemm_order(width: int) -> int:
    if width >= 64:
        return 64
    if width >= 32:
        return 32
    return 16  # Unused placeholder when P < 16.


def make_hadamard_gemm_matrix(
    order: int,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    if order not in GEMM_HADAMARD_ORDERS:
        raise ValueError(f"unsupported GEMM Hadamard order {order}")
    matrix = torch.empty((order, order), dtype=torch.float32)
    for row in range(order):
        for column in range(order):
            matrix[row, column] = 1.0 if (row & column).bit_count() % 2 == 0 else -1.0
    return matrix.to(device=device, dtype=dtype).contiguous()
