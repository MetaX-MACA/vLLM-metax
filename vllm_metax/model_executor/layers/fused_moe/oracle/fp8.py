# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import envs
from vllm.config.kernel import MoEBackend
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    logger,
    Fp8MoeBackend,
)

from vllm_metax.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)


def _get_priority_backends(
    moe_config: FusedMoEConfig,
    weight_key: QuantKey | None,
    activation_key: QuantKey | None,
) -> list[Fp8MoeBackend]:
    _AVAILABLE_BACKENDS = [
        Fp8MoeBackend.DEEPGEMM,
        Fp8MoeBackend.TRITON,
        Fp8MoeBackend.BATCHED_DEEPGEMM,
        Fp8MoeBackend.BATCHED_TRITON,
    ]
    return _AVAILABLE_BACKENDS


def backend_to_kernel_cls(
    backend: Fp8MoeBackend,
) -> list[type[mk.FusedMoEExperts]]:
    if backend == Fp8MoeBackend.DEEPGEMM:
        from vllm.model_executor.layers.fused_moe.experts.triton_deep_gemm_moe import (
            TritonOrDeepGemmExperts,
        )

        return [TritonOrDeepGemmExperts]

    elif backend == Fp8MoeBackend.BATCHED_DEEPGEMM:
        from vllm_metax.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe import (
            BatchedDeepGemmExperts,
        )

        return [BatchedDeepGemmExperts]

    elif backend == Fp8MoeBackend.TRITON:
        from vllm_metax.model_executor.layers.fused_moe.experts.triton_moe import (
            TritonExperts,
        )

        return [TritonExperts]

    elif backend == Fp8MoeBackend.BATCHED_TRITON:
        from vllm_metax.model_executor.layers.fused_moe.experts.fused_batched_moe import (
            BatchedTritonExperts,
        )

        return [BatchedTritonExperts]

    else:
        raise ValueError(f"Unknown FP8 MoE backend on Maca: {backend.value}")


def map_fp8_backend(runner_backend: MoEBackend) -> Fp8MoeBackend:
    """Map user's MoEBackend to Fp8MoeBackend."""
    mapping = {
        "triton": Fp8MoeBackend.TRITON,
        "deep_gemm": Fp8MoeBackend.DEEPGEMM,
    }
    if backend := mapping.get(runner_backend):
        return backend
    raise ValueError(
        f"moe_backend='{runner_backend}' is not supported for FP8 MoE. "
        f"Expected one of {list(mapping.keys())}."
    )


def select_fp8_moe_backend(
    config: FusedMoEConfig,
    weight_key: QuantKey | None,
    activation_key: QuantKey | None,
    allow_vllm_cutlass: bool = False,
) -> tuple[Fp8MoeBackend, type[mk.FusedMoEExperts] | None]:
    AVAILABLE_BACKENDS = _get_priority_backends(config, weight_key, activation_key)

    # NOTE(rob): We need to peak into the P/F selection to determine
    # if we are using the batched or standard expert format, which
    # if not ideal. Once we unify TP + DP/EP, we can select P/F first.
    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if config.moe_parallel_config.use_batched_activation_format
        else mk.FusedMoEActivationFormat.Standard
    )

    def _make_log_backend(backend: Fp8MoeBackend):
        available_backend_strs = [b.value for b in AVAILABLE_BACKENDS]
        return (
            f"Using {backend.value} Fp8 MoE backend out "
            f"of potential backends: {available_backend_strs}."
        )

    def _make_log_unsupported(backend: Fp8MoeBackend, reason: str | None) -> str:
        if reason:
            return (
                f"FP8 MoE backend {backend.value} does not support the "
                f"deployment configuration since {reason}."
            )
        else:
            return (
                f"FP8 MoE backend '{backend.value}' does not support the "
                "deployment configuration."
            )

    def _return_or_raise(
        backend: Fp8MoeBackend,
        config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[Fp8MoeBackend, type[mk.FusedMoEExperts]]:
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
        requested_backend = map_fp8_backend(runner_backend)
        # For batched activation format, use batched variants if available.
        if activation_format == mk.FusedMoEActivationFormat.BatchedExperts:
            if requested_backend == Fp8MoeBackend.DEEPGEMM:
                requested_backend = Fp8MoeBackend.BATCHED_DEEPGEMM
            elif requested_backend == Fp8MoeBackend.TRITON:
                requested_backend = Fp8MoeBackend.BATCHED_TRITON

    # Handle explicit DeepGEMM FP8 configuration.
    if not envs.is_set("VLLM_USE_DEEP_GEMM"):
        AVAILABLE_BACKENDS.remove(Fp8MoeBackend.DEEPGEMM)
        AVAILABLE_BACKENDS.remove(Fp8MoeBackend.BATCHED_DEEPGEMM)
    if envs.is_set("VLLM_USE_DEEP_GEMM") or envs.is_set("VLLM_MOE_USE_DEEP_GEMM"):
        if not envs.VLLM_USE_DEEP_GEMM or not envs.VLLM_MOE_USE_DEEP_GEMM:
            AVAILABLE_BACKENDS.remove(Fp8MoeBackend.DEEPGEMM)
            AVAILABLE_BACKENDS.remove(Fp8MoeBackend.BATCHED_DEEPGEMM)
        else:
            backend = (
                Fp8MoeBackend.DEEPGEMM
                if activation_format == mk.FusedMoEActivationFormat.Standard
                else Fp8MoeBackend.BATCHED_DEEPGEMM
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
                print(_make_log_unsupported(backend, reason))
                logger.debug_once(_make_log_unsupported(backend, reason))

    raise NotImplementedError(
        "No FP8 MoE backend on Maca supports the deployment configuration."
    )


def make_fp8_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    experts_cls: type[mk.FusedMoEExperts],
    fp8_backend: Fp8MoeBackend,
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
