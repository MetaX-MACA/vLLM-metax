# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Disable Kimi-K3's NVIDIA-only low-latency GEMM table on MACA. A device
#     capability match alone does not establish support for CuTe DSL/KDA kernels.
#     Returning no table keeps existing linear methods and projection-overlap
#     settings intact instead of installing NVIDIA plans during weight loading.
#
# Affected versions: vLLM 0.33.0
#     (4ba787ec9959e566d3b587a796dff07673324da5).
#
# Remove at: upstream gates the low-latency dispatch on a real CUDA backend
#     rather than device-capability aliases, or MACA provides equivalent
#     low-latency kernels and they are validated here.
# -----------------------------------------------------------------------------
"""Disable CUDA-only Kimi-K3 low-latency GEMMs on MACA."""

from vllm_metax.patch.utils import patch


@patch("vllm.models.kimi_k3.nvidia.low_latency_gemm", "_low_latency_table")
def _low_latency_table() -> None:
    # NOTE(metax): No NVIDIA projection table is supported on MACA.
    return None
