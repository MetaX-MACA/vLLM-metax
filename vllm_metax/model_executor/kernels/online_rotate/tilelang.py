# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

"""vLLM custom-op wrapper for the TileLang Hadamard kernel."""

from __future__ import annotations

import torch

from vllm.utils.torch_utils import direct_register_custom_op


def hadamard_tilelang(
    x: torch.Tensor,
    sign: torch.Tensor,
    hk: torch.Tensor,
    h_gemm: torch.Tensor,
    inverse: bool,
    k: int,
    p: int,
) -> torch.Tensor:
    """Execute the TileLang Hadamard transform."""
    if x.reshape(-1, x.shape[-1]).shape[0] == 0:
        return torch.empty_like(x)

    # Keep TileLang out of the model import path and compile on first use.
    from .tilelang_kernels import hadamard as hadamard_impl

    return hadamard_impl(
        x,
        sign,
        inverse=inverse,
        hk=hk,
        h_gemm=h_gemm,
        k=k,
        p=p,
    )


def _hadamard_tilelang_fake(
    x: torch.Tensor,
    sign: torch.Tensor,
    hk: torch.Tensor,
    h_gemm: torch.Tensor,
    inverse: bool,
    k: int,
    p: int,
) -> torch.Tensor:
    return torch.empty_like(x)


direct_register_custom_op(
    op_name="mx_hadamard_tilelang",
    op_func=hadamard_tilelang,
    mutates_args=[],
    fake_impl=_hadamard_tilelang_fake,
)


def hadamard(
    x: torch.Tensor,
    sign: torch.Tensor,
    hk: torch.Tensor,
    h_gemm: torch.Tensor,
    *,
    inverse: bool,
    k: int,
    p: int,
) -> torch.Tensor:
    """Apply the registered TileLang Hadamard custom op."""
    return torch.ops.vllm.mx_hadamard_tilelang(
        x,
        sign,
        hk,
        h_gemm,
        inverse,
        k,
        p,
    )


__all__ = ["hadamard", "hadamard_tilelang"]
