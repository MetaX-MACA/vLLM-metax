# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

from .tilelang import hadamard
from .utils import (
    factor_hadamard_dim,
    resolve_hadamard_dim,
    make_hadamard_gemm_matrix,
    select_gemm_order,
)
from .engram_gate import fused_engram_gate

__all__ = [
    "factor_hadamard_dim",
    "hadamard",
    "resolve_hadamard_dim",
    "make_hadamard_gemm_matrix",
    "select_gemm_order",
    "fused_engram_gate",
]
