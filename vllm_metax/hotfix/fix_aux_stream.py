# SPDX-License-Identifier: Apache-2.0
import torch


def _aux_stream() -> torch.cuda.Stream | None:
    """
    Ensures aux_stream is initialized only once
    """
    global _aux_stream

    from vllm.platforms import current_platform

    # TODO: validate this works properly on ROCm platform.
    if _aux_stream is None and current_platform.is_cuda_alike():
        _aux_stream = torch.cuda.Stream()

    return _aux_stream


from vllm.utils import torch_utils

torch_utils.aux_stream = _aux_stream
