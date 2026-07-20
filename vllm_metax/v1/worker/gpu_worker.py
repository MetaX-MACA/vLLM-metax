# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Maca GPU worker with optional msprobe precision-debug hooks.

Only selected by `MacaPlatformBase.check_and_update_config` when the user
requested a msprobe dump via `--additional-config`; the stock
`vllm.v1.worker.gpu_worker.Worker` is used otherwise, so this class adds no
overhead to the common path.
"""

from typing import Any

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput
from vllm.v1.worker.gpu_worker import Worker

from vllm_metax.utils.msprobe_debug import build_precision_debugger


class MacaWorker(Worker):
    """`Worker` variant that brackets every `execute_model` call with a
    msprobe `PrecisionDebugger` start/stop/step session."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._precision_debugger: Any | None = None

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        super().load_model(load_dummy_weights=load_dummy_weights)
        self._precision_debugger = build_precision_debugger(
            self.vllm_config.additional_config
        )

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | None:
        if self._precision_debugger is None:
            return super().execute_model(scheduler_output)

        self._precision_debugger.start(model=self.model_runner.model)
        try:
            return super().execute_model(scheduler_output)
        finally:
            self._precision_debugger.stop()
            self._precision_debugger.step()
