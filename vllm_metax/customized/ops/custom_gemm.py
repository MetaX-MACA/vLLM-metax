# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


def custom_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """BF16 GEMM accelerated by MACA C500 WMMA intrinsics.

    Computes ``out = a @ b`` where:
      - ``a``: [M, K] bfloat16, row-major
      - ``b``: [K, N] bfloat16, row-major
      - ``out``: [M, N] float32, **pre-allocated**

    The caller must allocate ``out`` with the correct shape and dtype
    before calling this function.  This is an in-place-output kernel;
    nothing is returned.

    Raises:
        ValueError: If shapes are inconsistent or dtypes are not
                    bfloat16 / float32.
    """
    # Lazy-load the source-built _C extension at first call.  During early
    # plugin init, import_kernels() may fail to load vllm_metax._C; by the
    # time user code calls this function, libtorch.so is available.
    import vllm_metax._C  # noqa: F401

    if a.dim() != 2 or b.dim() != 2:
        raise ValueError(
            f"custom_gemm: expected 2-D inputs, got a.shape={a.shape}, "
            f"b.shape={b.shape}"
        )
    if a.dtype != torch.bfloat16 or b.dtype != torch.bfloat16:
        raise ValueError(
            f"custom_gemm: inputs must be bfloat16, got "
            f"a.dtype={a.dtype}, b.dtype={b.dtype}"
        )
    if out.dtype != torch.float32:
        raise ValueError(
            f"custom_gemm: out must be float32, got out.dtype={out.dtype}"
        )

    M, K = a.shape
    K_b, N = b.shape

    if K != K_b:
        raise ValueError(
            f"custom_gemm: inner dimension mismatch: a is [M,K]=[{M},{K}], "
            f"b is [K,N]=[{K_b},{N}]"
        )
    if out.shape != (M, N):
        raise ValueError(
            f"custom_gemm: output shape mismatch, expected [{M},{N}], "
            f"got {list(out.shape)}"
        )

    torch.ops._C.custom_gemm(out, a, b, M, N, K)
