# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .kernels import chunk_kda_with_fused_gate, fused_recurrent_kda

__all__ = ["chunk_kda_with_fused_gate", "fused_recurrent_kda"]
