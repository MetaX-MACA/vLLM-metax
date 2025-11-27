# SPDX-License-Identifier: Apache-2.0

# -----------------------------------------------------
# Note: This is a hotfix to ensure the `mx_aux_stream`
#       is correctly initialized in maca platform by
#       *is_cuda_alike* check.
#
# TODO(hank): Remove this once the issue is resolved.
# -----------------------------------------------------
import torch

from vllm.utils.torch_utils import _aux_stream


def mx_aux_stream() -> torch.cuda.Stream | None:
    """
    Ensures aux_stream is initialized only once
    """
    global _aux_stream

    from vllm.platforms import current_platform

    # TODO: validate this works properly on ROCm platform.
    if _aux_stream is None and current_platform.is_cuda_alike():
        _aux_stream = torch.cuda.Stream()

    return _aux_stream


import vllm.model_executor.layers.fused_moe.layer

vllm.model_executor.layers.fused_moe.layer.aux_stream = mx_aux_stream

import vllm.utils.torch_utils

vllm.utils.torch_utils.aux_stream = mx_aux_stream
