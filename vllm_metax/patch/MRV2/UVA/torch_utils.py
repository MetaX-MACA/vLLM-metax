# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch
from collections.abc import Sequence
# ---------------------------------------------
# TODO(m01016): remove this file in v0.19.0.
# ---------------------------------------------


def get_accelerator_view_from_cpu_tensor(cpu_tensor: torch.Tensor) -> torch.Tensor:
    """
    Get an accelerator view of a CPU tensor using Unified Virtual Addressing (UVA).
    """
    from vllm.platforms import current_platform

    if current_platform.is_xpu():
        assert cpu_tensor.is_pinned(), "CPU tensor must be pinned"
        return torch.ops._C.get_xpu_view_from_cpu_tensor(cpu_tensor)
    elif current_platform.is_cuda_alike():
        return torch.ops._C.get_cuda_view_from_cpu_tensor(cpu_tensor)
    else:
        raise ValueError(
            f"`get_accelerator_view_from_cpu_tensor` is currently "
            f"not supported in: {current_platform.device_name}"
        )


import vllm.v1.worker.gpu.buffer_utils

vllm.v1.worker.gpu.buffer_utils.get_accelerator_view_from_cpu_tensor = (
    get_accelerator_view_from_cpu_tensor
)


import vllm.utils.torch_utils

vllm.utils.torch_utils.get_accelerator_view_from_cpu_tensor = (
    get_accelerator_view_from_cpu_tensor
)
