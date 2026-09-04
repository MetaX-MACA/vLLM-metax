# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


from enum import Enum

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config.kernel import MoEBackend
from vllm_metax.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt8DynamicTokenSym,
    kInt8StaticChannelSym,
)

from vllm_metax.patch.bugfix.int8_w8a8.int8_moe_config import (
    int8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.config import int8_w8a16_moe_quant_config

from vllm.model_executor.layers.fused_moe.oracle.int8 import (
    logger,
)

from vllm import envs


class Int8MoeBackend(Enum):
    TRITON = "TRITON"
    DEEPGEMM = "DEEPGEMM"
    BATCHED_DEEPGEMM = "BATCHED_DEEPGEMM"
    BATCHED_TRITON = "BATCHED_TRITON"


def _get_priority_backends(
    moe_config: FusedMoEConfig,
) -> list[Int8MoeBackend]:
    """
    Get available backends in priority order based on platform and config.
    """
    _AVAILABLE_BACKENDS = [
        Int8MoeBackend.TRITON,
        Int8MoeBackend.DEEPGEMM,
        Int8MoeBackend.BATCHED_TRITON,
        Int8MoeBackend.BATCHED_DEEPGEMM,
    ]

    return _AVAILABLE_BACKENDS


def backend_to_kernel_cls(
    backend: Int8MoeBackend,
) -> list[type[mk.FusedMoEExperts]]:
    if backend == Int8MoeBackend.TRITON:
        from vllm_metax.model_executor.layers.fused_moe.experts.triton_moe import (
            TritonExperts,
        )

        return [TritonExperts]
    elif backend == Int8MoeBackend.BATCHED_TRITON:
        from vllm_metax.model_executor.layers.fused_moe.experts.fused_batched_moe import (
            BatchedTritonExperts,
        )

        return [BatchedTritonExperts]
    elif backend == Int8MoeBackend.BATCHED_DEEPGEMM:
        from vllm_metax.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe import (
            BatchedDeepGemmExperts,
        )

        return [BatchedDeepGemmExperts]
    else:
        raise ValueError(f"Unsupported Int8 MoE backend: {backend.value}")


def map_int8_backend(runner_backend: MoEBackend) -> Int8MoeBackend:
    """Map user's MoEBackend to Int8MoeBackend."""
    mapping = {
        "triton": Int8MoeBackend.TRITON,
        "batched_triton": Int8MoeBackend.BATCHED_TRITON,
        "batched_deepgemm": Int8MoeBackend.BATCHED_DEEPGEMM,
    }
    if backend := mapping.get(runner_backend):
        return backend
    raise ValueError(
        f"moe_backend='{runner_backend}' is not supported for Int8 MoE. "
        f"Expected one of {list(mapping.keys())}."
    )


def select_int8_moe_backend(
    config: FusedMoEConfig,
    weight_key: QuantKey | None = kInt8StaticChannelSym,
    activation_key: QuantKey | None = kInt8DynamicTokenSym,
) -> tuple[Int8MoeBackend, type[mk.FusedMoEExperts]]:
    """
    Select the primary Int8 MoE backend.
    Note: Shape-specific fallbacks may still occur at runtime.
    """

    AVAILABLE_BACKENDS = _get_priority_backends(config)

    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if config.moe_parallel_config.use_batched_activation_format
        else mk.FusedMoEActivationFormat.Standard
    )

    def _make_log_backend(backend: Int8MoeBackend) -> str:
        available_backend_strs = [b.value for b in AVAILABLE_BACKENDS]
        return (
            f"Using {backend.value} Int8 MoE backend out "
            f"of potential backends: {available_backend_strs}."
        )

    def _make_log_unsupported(backend: Int8MoeBackend, reason: str | None) -> str:
        if reason:
            return (
                f"Int8 MoE backend {backend.value} does not support the "
                f"deployment configuration since {reason}."
            )
        else:
            return (
                f"Int8 MoE backend '{backend.value}' does not support the "
                "deployment configuration."
            )

    def _return_or_raise(
        backend: Int8MoeBackend,
        config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[Int8MoeBackend, type[mk.FusedMoEExperts]]:
        for k_cls in backend_to_kernel_cls(backend):
            supported, reason = k_cls.is_supported_config(
                k_cls, config, weight_key, activation_key, activation_format
            )
            if supported:
                logger.info_once(_make_log_backend(backend))
                return backend, k_cls
        raise ValueError(_make_log_unsupported(backend, reason))

    # Handle explicit moe_backend from user.
    runner_backend = config.moe_backend
    if runner_backend != "auto":
        requested_backend = map_int8_backend(runner_backend)
        # For batched activation format, use batched variants if available.
        if activation_format == mk.FusedMoEActivationFormat.BatchedExperts:
            if requested_backend == Int8MoeBackend.DEEPGEMM:
                requested_backend = Int8MoeBackend.BATCHED_DEEPGEMM
            elif requested_backend == Int8MoeBackend.TRITON:
                requested_backend = Int8MoeBackend.BATCHED_TRITON

        return _return_or_raise(
            requested_backend, config, weight_key, activation_key, activation_format
        )

    # Handle explicit DeepGEMM FP8 configuration.
    if not envs.is_set("VLLM_USE_DEEP_GEMM"):
        AVAILABLE_BACKENDS.remove(Int8MoeBackend.DEEPGEMM)
        AVAILABLE_BACKENDS.remove(Int8MoeBackend.BATCHED_DEEPGEMM)
    if envs.is_set("VLLM_USE_DEEP_GEMM") or envs.is_set("VLLM_MOE_USE_DEEP_GEMM"):
        if not envs.VLLM_USE_DEEP_GEMM or not envs.VLLM_MOE_USE_DEEP_GEMM:
            AVAILABLE_BACKENDS.remove(Int8MoeBackend.DEEPGEMM)
            AVAILABLE_BACKENDS.remove(Int8MoeBackend.BATCHED_DEEPGEMM)
        else:
            if activation_format != mk.FusedMoEActivationFormat.BatchedExperts:
                raise ValueError(
                    "Only batched activation format is supported for DeepGEMM backend, "
                    "Please make sure if your parallel config meet the requirement."
                )
            backend = (
                Int8MoeBackend.DEEPGEMM
                if activation_format == mk.FusedMoEActivationFormat.Standard
                else Int8MoeBackend.BATCHED_DEEPGEMM
            )
            return _return_or_raise(
                backend, config, weight_key, activation_key, activation_format
            )

    # Select kernels in order of backend.
    for backend in AVAILABLE_BACKENDS:
        for k_cls in backend_to_kernel_cls(backend):
            supported, reason = k_cls.is_supported_config(
                k_cls,
                config,
                weight_key,
                activation_key,
                activation_format,
            )
            if supported:
                logger.info_once(_make_log_backend(backend))
                return backend, k_cls
            else:
                logger.debug_once(_make_log_unsupported(backend, reason))

    raise NotImplementedError(
        "No Int8 MoE backend supports the deployment configuration "
        f"(weight_key={weight_key}, activation_key={activation_key}). "
        "Set `VLLM_LOGGING_LEVEL=DEBUG` to see per-backend unsupported reasons."
    )


def make_int8_moe_quant_config(
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    a1_scale: torch.Tensor | None = None,
    a2_scale: torch.Tensor | None = None,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    per_act_token_quant: bool = False,
    gemm1_alpha: float | None = None,
    gemm1_beta: float | None = None,
    gemm1_clamp_limit: float | None = None,
) -> FusedMoEQuantConfig:
    if (a1_scale is None) != (a2_scale is None):
        raise ValueError("a1_scale and a2_scale must both be provided or both be None")

    scales_absent = a1_scale is None and a2_scale is None

    if scales_absent and not per_act_token_quant:
        return int8_w8a16_moe_quant_config(
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            w1_zp=None,
            w2_zp=None,
            w1_bias=w1_bias,
            w2_bias=w2_bias,
            gemm1_alpha=gemm1_alpha,
            gemm1_beta=gemm1_beta,
            gemm1_clamp_limit=gemm1_clamp_limit,
        )

    # remove the int8_w8a16 logic since dynamic per token int8_w8a8
    # has no a1/a2 scale, and we want to avoid confusion
    return int8_w8a8_moe_quant_config(
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        per_act_token_quant=per_act_token_quant,
        gemm1_alpha=gemm1_alpha,
        gemm1_beta=gemm1_beta,
        gemm1_clamp_limit=gemm1_clamp_limit,
    )


def convert_to_int8_moe_kernel_format(
    int8_backend: Int8MoeBackend,
    w13: torch.Tensor,
    w2: torch.Tensor,
    layer: torch.nn.Module | None = None,
    w13_scale: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert INT8 MoE weights to backend-specific kernel format."""
    if int8_backend == Int8MoeBackend.TRITON:
        # TODO(Hank): prepare shape for triton experts
        pass
    elif int8_backend == Int8MoeBackend.BATCHED_TRITON:
        # TODO(Hank): prepare shape for batched triton experts
        pass
    elif int8_backend == Int8MoeBackend.BATCHED_DEEPGEMM:
        # TODO(Hank): prepare shape for batched deep_gemm experts
        pass
    return w13, w2


def make_int8_moe_kernel(
    int8_backend: Int8MoeBackend,
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    experts_cls: type[mk.FusedMoEExperts],
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> mk.FusedMoEKernel:
    # Create Prepare/Finalize.
    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=moe_quant_config,
        routing_tables=routing_tables,
        allow_new_interface=True,
        use_monolithic=issubclass(experts_cls, mk.FusedMoEExpertsMonolithic),
    )
    assert prepare_finalize is not None

    logger.info_once("Using %s", prepare_finalize.__class__.__name__)

    # Create Experts.
    if prepare_finalize.activation_format == mk.FusedMoEActivationFormat.BatchedExperts:
        max_num_tokens = prepare_finalize.max_num_tokens_per_rank()
        assert max_num_tokens is not None
        experts = experts_cls(
            moe_config=moe_config,
            quant_config=moe_quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=prepare_finalize.num_dispatchers(),
        )
    else:
        experts = experts_cls(
            moe_config=moe_config,
            quant_config=moe_quant_config,
        )

    kernel = mk.FusedMoEKernel(
        prepare_finalize,
        experts,
    )

    return kernel
