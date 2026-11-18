# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: MRV2 compute and output-copy streams can contend on the same MACA
#     hardware pipe. Request queue balancing once after the first decode output
#     copy is submitted, preserving the stream wait, D2H copy and completion
#     event. The patch is disabled by default. Enable it with
#     VLLM_METAX_ENABLE_MRV2_HWQ_BALANCING=1 and
#     VLLM_METAX_DISABLE_ASYNC_OUTPUT=0.
#
# Affected versions: vLLM v0.33.0 (GPUModelRunner.sample_tokens and AsyncOutput).
#
# Remove at: A supported MACA runtime no longer shows the same-pipe wait
#     regression with MRV2 asynchronous output and no balancing tag.
# -----------------------------------------------------------------------------
"""Opt-in MACA queue balancing for Model Runner V2 decode output."""

from contextvars import ContextVar
from functools import wraps

from vllm_metax import envs as mx_envs

_decode_runner = ContextVar("metax_mrv2_hwq_decode_runner", default=None)


def _is_enabled() -> bool:
    if not mx_envs.VLLM_METAX_ENABLE_MRV2_HWQ_BALANCING:
        return False
    if mx_envs.VLLM_METAX_DISABLE_ASYNC_OUTPUT:
        from vllm.logger import init_logger

        init_logger(__name__).warning(
            "MRV2 HWQ balancing requires VLLM_METAX_DISABLE_ASYNC_OUTPUT=0; "
            "skipping the patch because asynchronous output is disabled."
        )
        return False
    return True


if _is_enabled():
    import torch

    from vllm.logger import init_logger
    from vllm.v1.worker.gpu import model_runner
    from vllm_metax.patch.utils import patch

    logger = init_logger(__name__)
    original_sample_tokens = model_runner.GPUModelRunner.sample_tokens
    original_async_output = model_runner.AsyncOutput

    @patch("vllm.v1.worker.gpu.model_runner", "GPUModelRunner.sample_tokens")
    @wraps(original_sample_tokens)
    def _sample_tokens_with_decode_context(self, *args, **kwargs):
        # The original method clears this state before building AsyncOutput.
        state = self.execute_model_state
        runner = None
        if (
            state is not None
            and not getattr(self, "_metax_hwq_balanced_tagged", False)
            and not state.input_batch.has_prefill
        ):
            runner = self
        token = _decode_runner.set(runner)
        try:
            return original_sample_tokens(self, *args, **kwargs)
        finally:
            _decode_runner.reset(token)

    @patch("vllm.v1.worker.gpu.model_runner", "AsyncOutput")
    class _BalancedAsyncOutput(original_async_output):  # type: ignore[valid-type, misc]
        def __init__(self, *args, **kwargs):
            # Preserve the original wait_stream, D2H and copy_event submission.
            super().__init__(*args, **kwargs)
            runner = _decode_runner.get()
            if runner is None or getattr(runner, "_metax_hwq_balanced_tagged", False):
                return
            torch.cuda.nvtx.range_push("ActiveStreamsToHwQueues_Balanced")
            torch.cuda.nvtx.range_pop()
            runner._metax_hwq_balanced_tagged = True
            logger.info("MetaX HWQ balancing requested after decode output submission")
