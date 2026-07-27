# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config.kernel import MoEBackend
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)

from vllm_metax.utils.fused_moe import get_triton_experts_cls
from vllm.model_executor.layers.fused_moe.oracle.unquantized import (
    logger,
    UnquantizedMoeBackend,
)
from vllm_metax.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)


def _get_priority_backends(moe_config: FusedMoEConfig) -> list[UnquantizedMoeBackend]:
    _AVAILABLE_BACKENDS = [
        UnquantizedMoeBackend.TRITON,
        UnquantizedMoeBackend.BATCHED_TRITON,
    ]
    return _AVAILABLE_BACKENDS


def backend_to_kernel_cls(
    backend: UnquantizedMoeBackend,
) -> list[type[mk.FusedMoEExperts]]:
    if backend == UnquantizedMoeBackend.TRITON:
        TritonExperts = get_triton_experts_cls()
        return [TritonExperts]

    elif backend == UnquantizedMoeBackend.BATCHED_TRITON:
        from vllm_metax.model_executor.layers.fused_moe.experts.fused_batched_moe import (
            BatchedTritonExperts,
        )

        return [BatchedTritonExperts]

    else:
        raise ValueError(f"Unknown unquantized MoE backend on Maca: {backend.value}")


def map_unquantized_backend(runner_backend: MoEBackend) -> UnquantizedMoeBackend:
    """Map user's MoEBackend to UnquantizedMoeBackend."""
    mapping = {
        "triton": UnquantizedMoeBackend.TRITON,
    }
    if backend := mapping.get(runner_backend):
        return backend
    raise ValueError(
        f"moe_backend='{runner_backend}' is not supported for unquantized MoE. "
        f"Expected one of {list(mapping.keys())}."
    )


def select_unquantized_moe_backend(
    moe_config: FusedMoEConfig,
) -> tuple[UnquantizedMoeBackend, type[mk.FusedMoEExperts] | None]:
    """
    Select the primary Unquantized MoE backend.
    Note: Shape-specific fallbacks may still occur at runtime.
    """

    if moe_config.is_lora_enabled:
        logger.info_once("Using TRITON Unquantized MoE LoRA backend")
        return UnquantizedMoeBackend.TRITON, backend_to_kernel_cls(
            UnquantizedMoeBackend.TRITON
        )[0]

    # NOTE: the kernels are selected in the following order.
    AVAILABLE_BACKENDS = _get_priority_backends(moe_config)

    # NOTE(rob): We need to peak into the P/F selection to determine
    # if we are using the batched or standard expert format, which
    # if not ideal. Once we unify TP + DP/EP, we can select P/F first.
    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if moe_config.moe_parallel_config.use_batched_activation_format
        else mk.FusedMoEActivationFormat.Standard
    )

    def _make_log_backend(backend: UnquantizedMoeBackend) -> str:
        available_strs = [b.value for b in AVAILABLE_BACKENDS]
        return (
            f"Using {backend.value} Unquantized MoE backend out "
            f"of potential backends: {available_strs}."
        )

    def _make_log_unsupported(
        backend: UnquantizedMoeBackend, reason: str | None
    ) -> str:
        if reason:
            return (
                f"Unquantized MoE backend {backend.value} does not support the "
                f"deployment configuration since {reason}."
            )
        return (
            f"Unquantized MoE backend '{backend.value}' does not support the "
            "deployment configuration."
        )

    def _return_or_raise(
        backend: UnquantizedMoeBackend,
        config: FusedMoEConfig,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[UnquantizedMoeBackend, type[mk.FusedMoEExperts] | None]:
        reason = None
        for k_cls in backend_to_kernel_cls(backend):
            supported, reason = k_cls.is_supported_config(
                k_cls, config, None, None, activation_format
            )
            if supported:
                logger.info_once(_make_log_backend(backend))
                return backend, k_cls
        raise ValueError(_make_log_unsupported(backend, reason))

    runner_backend = moe_config.moe_backend
    # 'humming' is quantization-only; an unquantized layer (e.g. excluded via
    # modules_to_not_convert) falls through to auto instead of erroring.
    if runner_backend not in ["auto", "humming"]:
        requested_backend = map_unquantized_backend(runner_backend)
        if (
            activation_format == mk.FusedMoEActivationFormat.BatchedExperts
            and requested_backend == UnquantizedMoeBackend.TRITON
        ):
            requested_backend = UnquantizedMoeBackend.BATCHED_TRITON

        return _return_or_raise(requested_backend, moe_config, activation_format)

    for backend in AVAILABLE_BACKENDS:
        for k_cls in backend_to_kernel_cls(backend):
            supported, reason = k_cls.is_supported_config(
                k_cls, moe_config, None, None, activation_format
            )
            if supported:
                logger.info_once(_make_log_backend(backend))
                return backend, k_cls

            logger.debug_once(_make_log_unsupported(backend, reason))

    raise NotImplementedError(
        "No Unquantized MoE backend supports the deployment configuration."
    )


def make_unquantized_moe_kernel(
    quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    backend: UnquantizedMoeBackend,
    experts_cls: type[mk.FusedMoEExperts],
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> mk.FusedMoEKernel:
    # Create Prepare/Finalize
    is_monolithic = issubclass(experts_cls, mk.FusedMoEExpertsMonolithic)
    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=quant_config,
        routing_tables=routing_tables,
        allow_new_interface=True,
        use_monolithic=is_monolithic,
    )
    assert prepare_finalize is not None

    logger.info_once("Using %s", prepare_finalize.__class__.__name__)
    logger.info_once("Using %s MoE backend", experts_cls.__name__)

    # Create Experts
    if prepare_finalize.activation_format == mk.FusedMoEActivationFormat.BatchedExperts:
        max_num_tokens = prepare_finalize.max_num_tokens_per_rank()
        assert max_num_tokens is not None
        experts = experts_cls(
            moe_config=moe_config,
            quant_config=quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=prepare_finalize.num_dispatchers(),
        )
    else:
        experts = experts_cls(
            moe_config=moe_config,
            quant_config=quant_config,
        )

    kernel = mk.FusedMoEKernel(
        prepare_finalize,
        experts,
    )

    return kernel
