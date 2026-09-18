# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: This patch is fix DeepSeek-V4 trap when use pipeline parallel,
#       temporarily disable the optimization of use all_gather in send/recv
#
# Affected versions: vLLM 0.29.1.dev0 (98dff2a81d), verified 2026-09-17.
#
# Remove at: MACA pipeline send/recv safely supports the upstream all-gather
#     optimization.
# -----------------------------------------------------------------------------

from vllm_metax.patch.utils import patch

from vllm.distributed.parallel_state import GroupCoordinator


@patch("vllm.distributed.parallel_state", "GroupCoordinator._should_use_all_gather")
def _should_use_all_gather(
    self,
    key: str,
    numel: int,
    all_gather_group: "GroupCoordinator | None",
    all_gather_tensors: dict[str, bool] | None,
) -> bool:
    return False
