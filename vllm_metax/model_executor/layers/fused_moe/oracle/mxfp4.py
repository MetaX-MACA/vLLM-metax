# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
)
from vllm_metax.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.utils import replace_parameter
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
    Mxfp4MoeBackend,
    _backend_activation_key,
    _make_log_backend,
    _make_log_unsupported,
    _validate_mxfp4_experts_config,
)

logger = init_logger(__name__)


def backend_to_kernel_cls(
    backend: Mxfp4MoeBackend,
) -> list[type[mk.FusedMoEExperts]]:
    if backend == Mxfp4MoeBackend.TRITON:
        from vllm_metax.model_executor.layers.fused_moe.experts.triton_moe import (
            TritonExperts,
        )

        return [TritonExperts]

    else:
        raise ValueError(f"Not supported MXFP4 MoE backend: {backend.value}")


def _get_priority_backends() -> list[Mxfp4MoeBackend]:
    """Get available backends in priority order. SM100+ prefers DeepGEMM FP4 /
    TRTLLM MXFP8; SM90 falls through to Humming or Marlin (the
    backend-level ``is_supported_config`` check filters by device capability).
    """
    _AVAILABLE_BACKENDS = [
        Mxfp4MoeBackend.TRITON,
    ]
    return _AVAILABLE_BACKENDS


def select_deepseek_v4_mxfp4_moe_backend(
    config: FusedMoEConfig,
) -> tuple[Mxfp4MoeBackend, type[mk.FusedMoEExperts] | None]:
    """Select the MXFP4 MoE backend with MXFP8 activation as top priority.
    Falls back through BF16 and other backends.
    """
    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if config.moe_parallel_config.use_batched_activation_format
        else mk.FusedMoEActivationFormat.Standard
    )

    priority_backends = _get_priority_backends()

    # Iterate priority backends: TRTLLM MXFP8, then Triton.
    for backend in priority_backends:
        activation_key = _backend_activation_key(backend)
        for k_cls in backend_to_kernel_cls(backend):
            supported, reason = k_cls.is_supported_config(
                k_cls, config, kMxfp4Static, activation_key, activation_format
            )
            if supported:
                logger.info_once(_make_log_backend(backend), scope="local")
                return backend, k_cls
            else:
                logger.debug_once(_make_log_unsupported(backend, reason), scope="local")

    raise NotImplementedError(
        "No MXFP4 MoE backend supports the deployment configuration."
    )


def convert_weight_to_mxfp4_moe_kernel_format(
    mxfp4_backend: Mxfp4MoeBackend,
    layer: torch.nn.Module,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
    _cache_permute_indices: dict[torch.Size, torch.Tensor] | None = None,
    activation: MoEActivation | None = None,
    use_separated_a4w4: bool = False,
    activation_key: QuantKey | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    if mxfp4_backend == Mxfp4MoeBackend.TRITON:
        # The mctlass W4A16 GEMM consumes N-major block scales
        # (``[experts, K / 32, N]``), so transpose them here, as part of the
        # kernel-format conversion, instead of on every forward inside
        # ``mctlassEx_fused_moe_mxfp4_w4a16_gemm``. They are registered on the
        # layer so that the quantization config the experts read keeps aliasing
        # the layer's parameters.
        replace_parameter(
            layer,
            "w13_weight_scale",
            w13_weight_scale.transpose(1, 2).contiguous(),
        )
        replace_parameter(
            layer,
            "w2_weight_scale",
            w2_weight_scale.transpose(1, 2).contiguous(),
        )
        w13_weight_scale = layer.w13_weight_scale
        w2_weight_scale = layer.w2_weight_scale
        return (
            w13_weight,
            w2_weight,
            w13_weight_scale,
            w2_weight_scale,
            w13_bias,
            w2_bias,
        )
    else:
        raise ValueError(
            f"Unsupported mxfp4_backend for Mxfp4MoEMethod: {mxfp4_backend}. "
            f"Expected TRTLLM, Triton, AITER, XPU, or emulation backend."
        )


def make_mxfp4_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    experts_cls: type[mk.FusedMoEExperts],
    mxfp4_backend: Mxfp4MoeBackend,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> mk.FusedMoEKernel:
    """Create a FusedMoEKernel for the given MXFP4 backend."""
    _validate_mxfp4_experts_config(experts_cls, moe_config, mxfp4_backend)

    is_monolithic = issubclass(experts_cls, mk.FusedMoEExpertsMonolithic)

    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=moe_quant_config,
        routing_tables=routing_tables,
        allow_new_interface=True,
        use_monolithic=is_monolithic,
    )
    assert prepare_finalize is not None

    logger.info_once("Using %s", prepare_finalize.__class__.__name__)
    logger.info_once("Using %s", experts_cls.__name__)

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


def mxfp4_round_up_hidden_size_and_intermediate_size(
    backend: Mxfp4MoeBackend,
    hidden_size: int,
    intermediate_size: int,
    activation: MoEActivation | None = None,
) -> tuple[int, int]:
    """Round up hidden_size and intermediate_size based on backend requirements."""
    if backend == Mxfp4MoeBackend.TRITON:
        return hidden_size, intermediate_size
    else:
        # NOTE(metax): fall back to upstream impl
        from vllm.model_executor.layers.fused_moe.oracle.mxfp4 import (
            mxfp4_round_up_hidden_size_and_intermediate_size as vllm_mxfp4_round_up_hidden_size_and_intermediate_size,
        )

        return vllm_mxfp4_round_up_hidden_size_and_intermediate_size(
            backend, hidden_size, intermediate_size, activation
        )
