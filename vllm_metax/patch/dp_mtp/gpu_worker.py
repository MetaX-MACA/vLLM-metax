# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# fix dp + mtp hang
# https://github.com/vllm-project/vllm/pull/35243/changes

from vllm.v1.worker.gpu_worker import Worker


class MacaWorker(Worker):
    def execute_dummy_batch(self) -> None:
        num_tokens = getattr(self.model_runner, "uniform_decode_query_len", 1)
        self.model_runner._dummy_run(num_tokens, uniform_decode=True)


Worker.execute_dummy_batch = MacaWorker.execute_dummy_batch
