# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Disable the Model Runner V2 target AsyncOutput copy-stream path for
#     MetaX debugging. Output tensors are copied synchronously on the model
#     main stream so an extra output copy stream is not introduced. The patch
#     is installed only when VLLM_METAX_DISABLE_ASYNC_OUTPUT is truthy.
#
# Affected versions: vLLM v0.33.0
#     (4ba787ec9959e566d3b587a796dff07673324da5).
#
# Remove at: The upstream model runner supports disabling the target output
#     copy stream without monkey patching, or the MetaX output-copy issue is
#     fixed in upstream vLLM.
# -----------------------------------------------------------------------------
"""MetaX runtime patch for the Model Runner V2 AsyncOutput copy stream."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import torch

from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    ModelRunnerOutput,
)
from vllm.v1.worker.gpu.async_utils import AsyncOutput
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm_metax.patch.utils import patch
import vllm_metax.envs as mx_envs

if TYPE_CHECKING:
    from vllm.distributed.aux_output_connector.worker import PendingAuxOutput

_original_gpu_model_runner_init = GPUModelRunner.__init__
_original_async_output = AsyncOutput


class SynchronousAsyncOutput(AsyncModelRunnerOutput):
    """Drop-in AsyncOutput replacement that never creates a copy stream."""

    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampler_output: SamplerOutput,
        num_sampled_tokens: torch.Tensor,
        main_stream: torch.cuda.Stream,
        copy_stream: torch.cuda.Stream,
        check_ep_fault: bool,
        pending_aux_output: PendingAuxOutput | None,
    ) -> None:
        # The target implementation owns the complete v0.33 output conversion
        # and fault handling. Reusing the main stream removes the extra copy
        # stream without freezing those evolving behaviors in this patch.
        self._delegate = _original_async_output(
            model_runner_output=model_runner_output,
            sampler_output=sampler_output,
            num_sampled_tokens=num_sampled_tokens,
            main_stream=main_stream,
            copy_stream=main_stream,
            check_ep_fault=check_ep_fault,
            pending_aux_output=pending_aux_output,
        )

    def get_output(self) -> ModelRunnerOutput:
        return self._delegate.get_output()

    def get_kv_connector_output(self):
        return self._delegate.get_kv_connector_output()


@functools.wraps(_original_gpu_model_runner_init)
def _metax_gpu_model_runner_init(self: Any, *args: Any, **kwargs: Any) -> None:
    _original_gpu_model_runner_init(self, *args, **kwargs)

    # /-------------------- MetaX Modification --------------------\
    # AsyncPoolingOutput still needs its stream; the generative target output
    # path is fully synchronous after replacing AsyncOutput.
    if not getattr(self, "is_pooling_model", False):
        self.output_copy_stream = None
    # \-------------------- MetaX Modification --------------------/


if mx_envs.VLLM_METAX_DISABLE_ASYNC_OUTPUT:
    # The patch utility rejects duplicate target installation, so install each
    # target exactly once when this module is imported by the MetaX plugin.
    patch("vllm.v1.worker.gpu.async_utils", "AsyncOutput")(SynchronousAsyncOutput)
    patch("vllm.v1.worker.gpu.model_runner", "AsyncOutput")(SynchronousAsyncOutput)
    patch("vllm.v1.worker.gpu.model_runner", "GPUModelRunner.__init__")(
        _metax_gpu_model_runner_init
    )
