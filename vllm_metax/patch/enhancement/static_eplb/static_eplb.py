# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Hook both runner load boundaries and shared EPLB async/step boundaries.
#     StaticEplbRecorder owns operations; upstream owns loading and suppression.
#
# Affected versions: Reviewed vLLM v0.33.0 (4ba787ec9959).
#
# Remove at: Native EPLB provides static placement import/export lifecycle hooks.
# -----------------------------------------------------------------------------
"""Install the four upstream hooks needed for static EPLB."""

from collections.abc import Callable
from functools import wraps
from typing import Any

from vllm.distributed.eplb.eplb_state import EplbState
from vllm.v1.worker.gpu.model_runner import GPUModelRunner as GPUModelRunnerV2
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm_metax.patch.utils import patch

from .recorder import StaticEplbRecorder

_original_start_async_loop = EplbState.start_async_loop
_original_step = EplbState.step


@patch("vllm.distributed.eplb.eplb_state", "EplbState.start_async_loop")
@wraps(_original_start_async_loop)
def start_async_loop(
    self: EplbState,
    rank_mapping: dict[int, int] | None = None,
    is_profile: bool = False,
) -> None:
    # Called inside native load_model, before the recorder can attach to the state.
    if (
        getattr(self, "_metax_static_eplb_recorder", None) is not None
        or StaticEplbRecorder.is_enabled()
    ):
        return None
    return _original_start_async_loop(
        self, rank_mapping=rank_mapping, is_profile=is_profile
    )


def _wrap_load_model(original: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(original)
    def load_model(self: Any, *args: Any, **kwargs: Any) -> Any:
        if (args and args[0]) or kwargs.get("load_dummy_weights", False):
            return original(self, *args, **kwargs)
        recorder = StaticEplbRecorder.from_env(self.parallel_config)
        result = original(self, *args, **kwargs)
        if recorder is not None:
            recorder.attach(self.eplb_state)
        return result

    return load_model


load_model = patch("vllm.v1.worker.gpu_model_runner", "GPUModelRunner.load_model")(
    _wrap_load_model(GPUModelRunner.load_model)
)
load_model_v2 = patch("vllm.v1.worker.gpu.model_runner", "GPUModelRunner.load_model")(
    _wrap_load_model(GPUModelRunnerV2.load_model)
)


@patch("vllm.distributed.eplb.eplb_state", "EplbState.step")
@wraps(_original_step)
def step(
    self: EplbState,
    is_dummy: bool = False,
    is_profile: bool = False,
    log_stats: bool = False,
) -> None:
    recorder = getattr(self, "_metax_static_eplb_recorder", None)
    if recorder is None:
        return _original_step(
            self, is_dummy=is_dummy, is_profile=is_profile, log_stats=log_stats
        )
    return recorder.step(is_dummy=is_dummy, is_profile=is_profile, log_stats=log_stats)
