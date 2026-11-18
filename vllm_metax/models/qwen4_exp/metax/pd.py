# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate Qwen4Exp native NIXL P/D transfer contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def supports_nixl_transfer(vllm_config: VllmConfig) -> bool:
    transfer = vllm_config.kv_transfer_config
    return (
        transfer is not None
        and transfer.kv_connector == "NixlConnector"
        and not transfer.kv_connector_module_path
    )


def validate_pd_config(vllm_config: VllmConfig) -> None:
    """Reject invalid state transfer before loading checkpoint weights.

    NIXL 0.33 transfers attention, circular compression, GDN and replicated
    PLE cache owners together. Layerwise connectors need a separate
    implementation to synchronize recurrent state.
    """
    if (
        vllm_config.kv_transfer_config is None
        or vllm_config.kv_transfer_config.kv_connector is None
    ):
        return
    if not supports_nixl_transfer(vllm_config):
        raise ValueError("Qwen4Exp PD currently requires the native NixlConnector.")
    if vllm_config.kv_transfer_config.kv_buffer_device != "cuda":
        raise ValueError("Qwen4Exp NIXL transfer requires GPU-resident cache buffers.")
    if vllm_config.parallel_config.pipeline_parallel_size != 1:
        raise ValueError(
            "Qwen4Exp PLE state transfer requires pipeline_parallel_size=1."
        )
    if vllm_config.scheduler_config.disable_hybrid_kv_cache_manager:
        raise ValueError("Qwen4Exp NIXL transfer requires hybrid KV cache management.")
    speculative = vllm_config.speculative_config
    if speculative is not None:
        draft = speculative.draft_model_config
        draft_config = getattr(draft, "hf_config", None)
        # vLLM 0.33 NIXL includes MTP attention caches and excludes the
        # trailing speculative SSM scratch slots. Both endpoints must use
        # the same MTP configuration; NIXL's native handshake checks it.
        if (
            speculative.method != "mtp"
            or getattr(draft_config, "model_type", None) != "qwen4_exp_mtp"
            or "Qwen4ExpMTP" not in getattr(draft_config, "architectures", ())
            or speculative.parallel_drafting
        ):
            raise ValueError(
                "Qwen4Exp NIXL transfer supports sequential Qwen4Exp MTP only; "
                "configure the same draft model on both P and D."
            )
    from vllm.model_executor.layers.mamba.mamba_utils import get_conv_state_layout

    if get_conv_state_layout() != "DS":
        raise ValueError(
            "Qwen4Exp NIXL transfer requires VLLM_SSM_CONV_STATE_LAYOUT=DS "
            "before starting both P and D."
        )
