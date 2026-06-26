# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Custom element-wise addition operator: out = a + b.

This operator dispatches to a MACA-optimized CUDA kernel.
"""

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


def custom_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Element-wise addition of two tensors, optimized for MACA C500.

    Args:
        a: Input tensor (float32, float16, or bfloat16).
        b: Input tensor, same shape and dtype as ``a``.

    Returns:
        A new tensor containing ``a + b``, same shape and dtype as inputs.
    """
    # Lazy-load the source-built _C extension at first call.  import_kernels()
    # can fail during early plugin init with ModuleNotFoundError for
    # vllm_metax._C; doing it here ensures libtorch.so is already available.
    import vllm_metax._C  # noqa: F401 — registers custom_add into torch.ops._C

    if a.shape != b.shape:
        raise ValueError(
            f"custom_add: shape mismatch: a.shape={a.shape} vs b.shape={b.shape}"
        )
    out = torch.empty_like(a)
    logger.info(
        "[CUSTOM_ADD] Python wrapper called: a.shape=%s, a.dtype=%s, a.device=%s",
        a.shape, a.dtype, a.device,
    )
    torch.ops._C.custom_add(out, a, b)
    return out
