# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Kimi-K3 GEMM-RS/AR requires NVIDIA CuTe DSL and NVLink multicast.
#     Disable its initializer before it imports unsupported CUDA kernels.
#
# Affected versions: vLLM 0.33.0 (4ba787ec9959).
#
# Remove at: Upstream gates this initializer on backend support, or equivalent
#     MACA kernels and collectives are available and validated.
# -----------------------------------------------------------------------------
"""Disable unsupported Kimi-K3 fused GEMM collectives on MACA."""

from vllm_metax.patch.utils import patch


@patch("vllm.models.kimi_k3.nvidia.model", "maybe_init_gemm_rs_ar")
def _maybe_init_gemm_rs_ar(vllm_config, use_sequence_parallel: bool) -> bool:
    return False
