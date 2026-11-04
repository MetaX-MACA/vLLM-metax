# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Disable the send/recv all-gather optimization for the recorded MACA PP
#     trap observed with DeepSeek-V4. This replaces GroupCoordinator globally,
#     so the workaround applies to every model using this communication path.
#
# Affected versions: vLLM 0.30.0 (ced6857afa0e), reviewed 2026-10-12.
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
