# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# ------------------------------------------------------------
# Note: This patch is to relax platform check at SMControlContextManager.__init__,
#       since MACA is CUDA-like and compatible with following code
# ------------------------------------------------------------

from vllm_metax.patch.utils import patch

from typing import Callable
import torch
from vllm.platforms import current_platform
from vllm.utils.platform_utils import num_compute_units


@patch("vllm.v1.worker.gpu_ubatch_wrapper", "SMControlContextManager.__init__")
def __init__(
    self,
    comm_sms: int,
    set_comm_sms: Callable[[int], None],
    set_compute_sms: Callable[[int], None],
):
    """
    Context manager for controlling SM (Streaming Multiprocessor)
    allocation. Upon entering the context, it sets the number of SMs
    allocated for communication and computation to comm_sms and
    total_sms - comm_sms respectively. Upon exiting, it restores the
    allocation to use all available SMs (i.e. total_sms).

    Args:
        comm_sms (int): The number of SMs to allocate for communication.
            (The remainder will be used for computation.)
        set_comm_sms (Callable[[int], None]):
            A function that sets the number of SMs for communication.
        set_compute_sms (Callable[[int], None]):
            A function that sets the number of SMs for computation.
    """
    # /-------------------- MetaX Modification --------------------\
    # MACA is CUDA-like and compatible with following code

    assert current_platform.is_cuda_alike() or current_platform.is_rocm(), (
        "SM/CU control is supported on CUDA-like and ROCm platforms"
    )
    # /-------------------- MetaX Modification --------------------\

    device = torch.accelerator.current_device_index()
    total_sms = num_compute_units(device)

    assert comm_sms < total_sms
    self.total_sms = total_sms
    self.compute_sms = total_sms - comm_sms
    self.comm_sms = comm_sms
    self.set_comm_sms = set_comm_sms
    self.set_compute_sms = set_compute_sms
