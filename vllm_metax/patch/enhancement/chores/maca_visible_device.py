# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Add `MACA_VISIBLE_DEVICES` handling alongside `CUDA_VISIBLE_DEVICES`.
#
# Affected versions: vLLM 0.29.1.dev0 (98dff2a81d), verified 2026-09-17.
#
# Remove at: Upstream worker launch propagates MACA_VISIBLE_DEVICES natively and
#     preserves explicitly supplied visibility settings.
# -----------------------------------------------------------------------------

from vllm.utils.system_utils import update_environment_variables
from vllm_metax.patch.utils import patch


@patch(
    "vllm.v1.worker.worker_base",
    "WorkerWrapperBase.update_environment_variables",
)
def update_environment_variables_with_maca(
    self, envs_list: list[dict[str, str]]
) -> None:
    envs = envs_list[self.rpc_rank].copy()
    key = "CUDA_VISIBLE_DEVICES"
    # /------------------------  Metax Modification -------------------------\
    if key in envs:
        envs.setdefault("MACA_VISIBLE_DEVICES", envs[key])
    # \------------------------- Metax Modification -------------------------/
    update_environment_variables(envs)
