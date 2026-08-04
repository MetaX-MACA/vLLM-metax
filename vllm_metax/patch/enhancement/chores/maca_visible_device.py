# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# -----------------------------------------------
# Note: Add `MACA_VISIBLE_DEVICES` handling alongside `CUDA_VISIBLE_DEVICES`.
#
# Affected versions: v0.21.0
# -----------------------------------------------

from vllm.utils.system_utils import update_environment_variables
from vllm_metax.patch.utils import patch


# ----------------------------------------------------
# TODO(hank): need to check vllm PR#33308 to see
# if this patch is still needed after the PR is merged.
@patch(
    "vllm.v1.worker.worker_base",
    "WorkerWrapperBase.update_environment_variables",
)
def update_environment_variables_with_maca(
    self, envs_list: list[dict[str, str]]
) -> None:
    envs = envs_list[self.rpc_rank]
    key = "CUDA_VISIBLE_DEVICES"
    # /------------------------  Metax Modification -------------------------\
    envs["MACA_VISIBLE_DEVICES"] = envs.get(key, "")
    # \------------------------- Metax Modification -------------------------/
    update_environment_variables(envs)
