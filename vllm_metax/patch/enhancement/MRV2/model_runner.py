# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Disable the Model Runner V2 target AsyncOutput copy-stream path for
#     MetaX debugging. Output tensors are copied synchronously on the model
#     main stream so an extra output copy stream is not introduced. The patch
#     is installed only when VLLM_METAX_DISABLE_ASYNC_OUTPUT is truthy.
#
# Affected versions: vLLM v0.30.0, verified 2026-10-23.
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

import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    get_ep_all2all_manager,
)
from vllm.v1.outputs import (
    AsyncModelRunnerOutput,
    ModelRunnerOutput,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.utils import raise_if_nan_logits
from vllm_metax.patch.utils import patch
import vllm_metax.envs as mx_envs

if TYPE_CHECKING:
    from vllm.distributed.aux_output_connector.worker import PendingAuxOutput

_original_gpu_model_runner_init = GPUModelRunner.__init__


def _copy_prompt_logprobs(value: Any) -> Any:
    if value is None:
        return None
    cu_num_generated_tokens_tensor = value.cu_num_generated_tokens_tensor
    if cu_num_generated_tokens_tensor is not None:
        cu_num_generated_tokens_tensor = cu_num_generated_tokens_tensor.cpu()
    return value._replace(
        logprob_token_ids=value.logprob_token_ids.cpu(),
        logprobs=value.logprobs.cpu(),
        selected_token_ranks=value.selected_token_ranks.cpu(),
        cu_num_generated_tokens_tensor=cu_num_generated_tokens_tensor,
    )


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
        self.model_runner_output = model_runner_output
        self.sampler_output = sampler_output
        self.num_sampled_tokens = num_sampled_tokens
        self.main_stream = main_stream
        self.check_ep_fault = check_ep_fault
        self.pending_aux_output = pending_aux_output
        # copy_stream is accepted only for compatibility with the upstream
        # AsyncOutput constructor and is intentionally unused.

    def get_output(self) -> ModelRunnerOutput:
        # /-------------------- MetaX Modification --------------------\
        # Use the model main stream instead of a dedicated copy stream.
        self.main_stream.synchronize()
        # \-------------------- MetaX Modification --------------------/

        sampled_token_ids = self.sampler_output.sampled_token_ids.cpu().tolist()
        num_sampled_tokens = self.num_sampled_tokens.cpu().numpy()
        for token_ids, num_tokens in zip(
            sampled_token_ids, num_sampled_tokens, strict=False
        ):
            del token_ids[int(num_tokens) :]
        self.model_runner_output.sampled_token_ids = sampled_token_ids

        if self.sampler_output.sampling_mask_tensors is not None:
            self.model_runner_output.sampling_masks = (
                self.sampler_output.sampling_mask_tensors.tolists()
            )

        if self.sampler_output.num_nans is not None:
            self.model_runner_output.num_nans_in_logits = dict(
                zip(
                    self.model_runner_output.req_ids,
                    self.sampler_output.num_nans.cpu().tolist(),
                    strict=False,
                )
            )
            if envs.VLLM_RAISE_ON_LOGIT_NANS:
                raise_if_nan_logits(self.model_runner_output.num_nans_in_logits)

        if self.sampler_output.logprobs_tensors is not None:
            self.model_runner_output.logprobs = (
                self.sampler_output.logprobs_tensors.tolists()
            )

        self.model_runner_output.prompt_logprobs_dict = {
            req_id: _copy_prompt_logprobs(value)
            for req_id, value in self.model_runner_output.prompt_logprobs_dict.items()
        }
        if self.pending_aux_output is not None:
            self.model_runner_output.aux_output_connector_output = (
                self.pending_aux_output.process_output()
            )

        if self.check_ep_fault:
            has_fault = get_ep_all2all_manager().query_fault()
            if has_fault.item():
                mask = get_ep_all2all_manager().query_active_mask()
                raise RuntimeError(
                    "Fault detected in EP all2all communication: "
                    "one or more ranks timed out during dispatch/combine. "
                    f"Mask: {mask.cpu().tolist()}"
                )

        return self.model_runner_output


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
