# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp implementation for the MetaX model registry."""

from .common.hyperconnection import (
    GatedResidual,
    GroupedGemmaRMSNorm,
    HyperConnectionBase,
    HyperConnectionConfig,
)

__all__ = [
    "GatedResidual",
    "GroupedGemmaRMSNorm",
    "HyperConnectionBase",
    "HyperConnectionConfig",
    "Qwen4ExpForCausalLM",
    "Qwen4ExpForConditionalGeneration",
    "Qwen4ExpMTP",
]


def __getattr__(name: str):
    if name == "Qwen4ExpMTP":
        from .metax.mtp import Qwen4ExpMTP

        return Qwen4ExpMTP
    if name in {"Qwen4ExpForCausalLM", "Qwen4ExpForConditionalGeneration"}:
        from .metax import model

        return getattr(model, name)
    raise AttributeError(name)
