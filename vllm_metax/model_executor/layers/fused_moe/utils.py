# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch
import importlib
from typing import Any

import vllm_metax.envs as mx_envs
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
)

_mctlass_modname = (
    "vllm_metax.model_executor.layers.quantization._python_api_ops"
    if mx_envs.MACA_VLLM_ENABLE_MCTLASS_PYTHON_API
    else "vllm_metax.model_executor.layers.quantization._cutlass_ops"
)
mctlass_ops: Any = importlib.import_module(_mctlass_modname)


def initialize_staged_config(
    config: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    # Metax Modification: prepare staged config for MACA
    stage1_config = config.get("stage1", config).copy()
    stage1_config.setdefault("SPLIT_K", 1)
    stage1_config.pop("ACCF32", None)

    stage2_config = config.get("stage2", config).copy()
    stage2_config.setdefault("SPLIT_K", 1)
    stage2_config.pop("ACCF32", None)
    return stage1_config, stage2_config


def get_config_dtype_str(
    dtype: torch.dtype,
    use_int4_w4a16: bool | None = False,
    # ┌------------------------  Metax Modification -------------------------┐
    use_int4_w4a8: bool | None = False,
    use_int8_w8a8: bool | None = False,
    # └------------------------- Metax Modification -------------------------┘
    use_int8_w8a16: bool | None = False,
    use_fp8_w8a8: bool | None = False,
    ocp_mx_scheme: str | None = None,
) -> str | None:
    if use_fp8_w8a8:
        return "fp8_w8a8"
    # ┌------------------------  Metax Modification -------------------------┐
    elif use_int4_w4a8:
        return "int4_w4a8"
    elif use_int8_w8a8:
        return "int8_w8a8"
    # └------------------------- Metax Modification -------------------------┘
    elif use_int8_w8a16:
        return "int8_w8a16"
    elif use_int4_w4a16:
        return "int4_w4a16"
    elif ocp_mx_scheme is not None:
        # The output of this function is passed to `try_get_optimal_moe_config`,
        # and as we only simulate OCP MX execution in fused_moe for now,
        # we will NOT look for `*,dtype=w_mxfp4_a_mxfp4.json` for now.
        return None
    elif dtype == torch.float:
        # avoiding cases where kernel fails when float32 MoE
        # use fp16/bfloat16 configs
        return "float32"
    return None


def _filter_triton_kernel_config(
    config: dict[str, Any],
    *,
    allowed_keys: set[str],
) -> dict[str, Any]:
    """Drop any stale/unknown keys from tuned configs before Triton launch.

    Some tuned JSON configs may contain historical fields that are not part of
    the Triton kernel signature / launch kwargs. Forwarding them would raise at
    runtime (unexpected keyword argument).
    """

    return {k: v for k, v in config.items() if k in allowed_keys}


def maybe_override_stage_block_size_m(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    intermediate_cache13: torch.Tensor,
    quant_config: FusedMoEQuantConfig,
    staged_configs: tuple[dict[str, Any], dict[str, Any]],
    top_k_num: int,
    block_shape: list[int] | None,
    num_tokens: int,
    N: int,
    K: int,
) -> None:
    if quant_config.use_int8_w8a8 and mx_envs.MACA_VLLM_ENABLE_MCTLASS_FUSED_MOE:
        kernel_m = mctlass_ops.cutlass_moe_mm_w8a8_get_kernel_m(
            hidden_states, w1, intermediate_cache13, top_k_num
        )
        assert kernel_m > 0, "cutlass_moe_w8a8 BLOCK_SIZE_M must greater than zero."
        staged_configs[0]["BLOCK_SIZE_M"] = kernel_m
        staged_configs[1]["BLOCK_SIZE_M"] = kernel_m

    if quant_config.use_int4_w4a8 and mx_envs.MACA_VLLM_ENABLE_MCTLASS_PYTHON_API:
        if block_shape is None:
            kernel_m = mctlass_ops.cutlass_moe_mm_w4a8_get_kernel_m_per_channel(
                a=hidden_states,
                b=w1,
                c=intermediate_cache13,
                K=K,
                num_valid_tokens=hidden_states.size(0) * top_k_num,
                topk=top_k_num,
            )
        else:
            kernel_m = mctlass_ops.mctlassEx_fused_moe_w4a8_get_kernel_m(
                hidden_states,
                w1.view(dtype=torch.quint4x2),
                intermediate_cache13,
                num_experts=w1.size(0),
                batch_size=hidden_states.size(0),
                N=N,
                K=K,
                num_valid_tokens=num_tokens,
                topk=top_k_num,
                group_size=block_shape[1],
            )
        assert kernel_m > 0, (
            "cutlass_fused_moe_w4a8 BLOCK_SIZE_M must greater than zero."
        )
        staged_configs[0]["BLOCK_SIZE_M"] = kernel_m
        staged_configs[1]["BLOCK_SIZE_M"] = kernel_m

    if (
        hidden_states.dtype == torch.bfloat16
        and not quant_config.use_int4_w4a8
        and not quant_config.use_int4_w4a16
        and not quant_config.use_int8_w8a8
        and not quant_config.use_int8_w8a16
        and mx_envs.MACA_VLLM_ENABLE_MCTLASS_FUSED_MOE
        and mx_envs.MACA_VLLM_ENABLE_MCTLASS_PYTHON_API
    ):
        kernel_m = mctlass_ops.mctlassEx_fused_moe_bf16_get_kernel_m(
            hidden_states,  # A
            w1,  # B
            intermediate_cache13,  # C
            w1.size(0),  # num_experts
            hidden_states.shape[0],  # batch_size
            N,  # N
            hidden_states.shape[1],  # K
            top_k_num,  # topk
        )
        assert kernel_m > 0, (
            "cutlass_fused_moe_bf16 BLOCK_SIZE_M must greater than zero."
        )
        staged_configs[0]["BLOCK_SIZE_M"] = kernel_m
        staged_configs[1]["BLOCK_SIZE_M"] = kernel_m
