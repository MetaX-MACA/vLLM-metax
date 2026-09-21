# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
)

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config.kernel import MoEBackend
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm_metax.model_executor.layers.fused_moe.experts.triton_moe import (
    TritonWNA16Experts,
)

from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
)
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import WNA16MoEBackend

logger = init_logger(__name__)


def backend_to_kernel_cls(
    backend: WNA16MoEBackend,
) -> list[type[mk.FusedMoEExperts]]:
    """Return the experts class for the given backend, or None for NONE."""
    if backend == WNA16MoEBackend.TRITON:
        return [TritonWNA16Experts]
    else:
        raise ValueError(f"Unknown WNA16 MoE backend: {backend.value}")


def _get_priority_backends() -> list[WNA16MoEBackend]:
    """
    Get available backends in priority order based on platform and config.
    """
    return [
        WNA16MoEBackend.TRITON,
    ]


def _backend_incompatibility_reason(
    backend: WNA16MoEBackend,
    moe_config: FusedMoEConfig,
    quant_config: QuantizationConfig | QuantizationArgs,
    may_have_zp: bool,
    may_have_bias: bool,
    allow_tile_padding: bool,
) -> str | None:
    from vllm.model_executor.layers.quantization.auto_awq import AutoAWQConfig
    from vllm.model_executor.layers.quantization.auto_gptq import AutoGPTQConfig

    if backend == WNA16MoEBackend.TRITON:
        if may_have_bias:
            return "expert bias is not supported"
        if isinstance(quant_config, AutoAWQConfig):
            return "the AutoAWQ weight layout is not supported"
        if isinstance(quant_config, AutoGPTQConfig) and quant_config.desc_act:
            return "GPTQ activation ordering is not supported"
        if (
            isinstance(quant_config, QuantizationArgs)
            and quant_config.actorder == "group"
        ):
            return "group activation ordering is not supported"

    return None


def map_wna16_backend(runner_backend: MoEBackend) -> WNA16MoEBackend:
    """Map user's MoEBackend to WNA16MoEBackend."""
    mapping = {
        "triton": WNA16MoEBackend.TRITON,
    }
    if backend := mapping.get(runner_backend):
        return backend
    raise ValueError(
        f"moe_backend='{runner_backend}' is not supported for WNA16 MoE. "
        f"Expected one of {list(mapping.keys())}."
    )


def select_wna16_moe_backend(
    config: FusedMoEConfig,
    weight_key: QuantKey,
    quant_config: QuantizationConfig | QuantizationArgs,
    may_have_zp: bool,
    may_have_bias: bool,
    allow_tile_padding: bool = False,
) -> tuple[WNA16MoEBackend, type[mk.FusedMoEExperts]]:
    """Select the WNA16 MoE backend.

    Args:
        config: the shared ``FusedMoEConfig`` for this layer.
        weight_key: The QuantKey describing the weight quantization.
                    Must have int4 or int8 type.
        quant_config: Quantization structure and checkpoint format description.
        may_have_zp: Whether the integration can provide weight zero points.
        may_have_bias: Whether the integration can provide expert bias.

    Returns:
        A tuple of (``WNA16MoEBackend``, experts class or ``None``).
    """

    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if config.moe_parallel_config.use_batched_activation_format
        else mk.FusedMoEActivationFormat.Standard
    )

    def _make_log_backend(backend: WNA16MoEBackend):
        return f"Using '{backend.value}' WNA16 MoE backend."

    def _make_log_unsupported(backend: WNA16MoEBackend, reason: str | None) -> str:
        if reason:
            return (
                f"WNA16 MoE backend '{backend.value}' does not support the "
                f"deployment configuration since {reason}."
            )
        return (
            f"WNA16 MoE backend '{backend.value}' does not support the "
            "deployment configuration."
        )

    def _return_or_raise(
        backend: WNA16MoEBackend,
        config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[WNA16MoEBackend, type[mk.FusedMoEExperts]]:
        reason: str | None = None
        for k_cls in backend_to_kernel_cls(backend):
            supported, reason = k_cls.is_supported_config(
                k_cls, config, weight_key, activation_key, activation_format
            )
            if supported:
                logger.info_once(_make_log_backend(backend), scope="local")
                return backend, k_cls
        raise ValueError(_make_log_unsupported(backend, reason))

    # Handle explicit moe_backend from user.
    runner_backend = config.moe_backend
    if runner_backend != "auto":
        requested_backend = map_wna16_backend(runner_backend)
        reason = _backend_incompatibility_reason(
            requested_backend,
            config,
            quant_config,
            may_have_zp,
            may_have_bias,
            allow_tile_padding,
        )
        if reason is not None:
            raise ValueError(_make_log_unsupported(requested_backend, reason))
        return _return_or_raise(
            requested_backend, config, weight_key, None, activation_format
        )

    # Select kernels in order of backend.
    AVAILABLE_BACKENDS = _get_priority_backends()

    for backend in AVAILABLE_BACKENDS:
        reason = _backend_incompatibility_reason(
            backend,
            config,
            quant_config,
            may_have_zp,
            may_have_bias,
            allow_tile_padding,
        )
        if reason is not None:
            logger.debug_once(_make_log_unsupported(backend, reason), scope="local")
            continue
        activation_key = None  # always BF16 activation for WNA16 MoE
        for k_cls in backend_to_kernel_cls(backend):
            supported, reason = k_cls.is_supported_config(
                k_cls, config, weight_key, activation_key, activation_format
            )
            if supported:
                logger.info_once(_make_log_backend(backend), scope="local")
                return backend, k_cls
            else:
                logger.debug_once(_make_log_unsupported(backend, reason), scope="local")

    raise NotImplementedError(
        "No WNA16 MoE backend supports the deployment configuration."
    )


def make_wna16_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    experts_cls: type[mk.FusedMoEExperts],
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
) -> mk.FusedMoEKernel:
    from vllm.model_executor.layers.fused_moe.all2all_utils import (
        maybe_make_prepare_finalize,
    )

    allowed_experts: tuple[type[mk.FusedMoEExperts], ...] = (TritonWNA16Experts,)

    assert experts_cls in allowed_experts

    is_monolithic = experts_cls.is_monolithic()

    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=moe_quant_config,
        routing_tables=routing_tables,
        allow_new_interface=True,
        use_monolithic=is_monolithic,
    )
    assert prepare_finalize is not None

    logger.info_once("Using %s", prepare_finalize.__class__.__name__, scope="local")
    logger.info_once("Using %s", experts_cls.__name__, scope="local")

    extra_args: dict[str, Any] = {}

    if prepare_finalize.activation_format == mk.FusedMoEActivationFormat.BatchedExperts:
        max_num_tokens = prepare_finalize.max_num_tokens_per_rank()
        assert max_num_tokens is not None
        extra_args["max_num_tokens"] = max_num_tokens
        extra_args["num_dispatchers"] = prepare_finalize.num_dispatchers()

    experts = experts_cls(
        moe_config=moe_config,
        quant_config=moe_quant_config,
        **extra_args,
    )

    return mk.FusedMoEKernel(
        prepare_finalize,
        experts,
    )


def convert_to_wna16_moe_kernel_format(
    backend: WNA16MoEBackend,
    layer: torch.nn.Module,
    quant_config: QuantizationConfig | QuantizationArgs | None,
    input_dtype: torch.dtype | None,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_g_idx: torch.Tensor | None = None,
    w2_g_idx: torch.Tensor | None = None,
    w13_qzeros: torch.Tensor | None = None,
    w2_qzeros: torch.Tensor | None = None,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> (
    tuple[
        torch.Tensor,  # w13_qweight
        torch.Tensor,  # w2_qweight
        torch.Tensor,  # w13_scales
        torch.Tensor,  # w2_scales
        torch.Tensor | None,  # w13_g_idx
        torch.Tensor | None,  # w2_g_idx
        torch.Tensor | None,  # w13_g_idx_sort_indices
        torch.Tensor | None,  # w2_g_idx_sort_indices
        torch.Tensor | None,  # w13_qzeros
        torch.Tensor | None,  # w2_qzeros
        torch.Tensor | None,  # w13_input_global_scale
        torch.Tensor | None,  # w2_input_global_scale
        torch.Tensor | None,  # w13_bias
        torch.Tensor | None,  # w2_bias
    ]
    | None
):
    """Dispatch weight post-processing to the appropriate per-backend handler.

    To add a new backend, implement a ``_process_weights_<name>`` helper and
    add a branch here. Backends that rewrite the layer's parameters in place
    (e.g. Humming) return ``None``; the caller then skips the param scatter.

    Args:
        backend: the selected ``WNA16MoEBackend``.
        layer: the ``MoERunner`` layer whose parameters are being prepared.
        quant_config: the ``QuantizationConfig`` for this layer.
        input_dtype: optional activation dtype, usually should be 16 bit.
    """
    if backend == WNA16MoEBackend.TRITON:
        # Two possible input layouts depending on the quantization source:
        #
        # MoeWNA16 (uint8):              (E, N_out, K // bit8_pack)  — N-first
        #   → just view as uint8 (no-op)
        #
        # AutoGPTQ/compressed-tensors (int32, K-first):
        #   (E, K // pack32, N_out)
        #   → transpose to N-first, then view as uint8 to get
        #     (E, N_out, K // bit8_pack)  [int32 = 4 bytes → 4 uint8s]
        #   Scales: (E, K // gs, N_out) → transpose → (E, N_out, K // gs)
        from vllm_metax.registry.quant_config.auto_gptq import (
            MacaAutoGPTQConfig,
        )

        if isinstance(quant_config, (MacaAutoGPTQConfig, QuantizationArgs)):
            # These integrations build in K-first format even when the Triton
            # backend is selected. Transpose to N-first first.
            w13_uint8 = w13.transpose(1, 2).contiguous().view(torch.uint8)
            w2_uint8 = w2.transpose(1, 2).contiguous().view(torch.uint8)
            w13_scale = w13_scale.transpose(1, 2).contiguous()
            w2_scale = w2_scale.transpose(1, 2).contiguous()
            # Zero points from compressed-tensors checkpoints are K-first int32
            # with 8 int4 ZPs packed per element: shape (E, K//gs, N//8).
            # fused_moe_kernel_gptq_awq expects N-first uint8 with 2 int4 ZPs
            # per byte: shape (E, N//2, K//gs), indexed as
            # (offs_bn // 2) * stride_bzn + offs_k_group * stride_bzk.
            # Expand the packed output-channel dimension into bytes BEFORE
            # transposing: (E, K//gs, N//8) int32 -> (E, K//gs, N//2)
            # uint8 -> (E, N//2, K//gs). With a single K group, transposing
            # first leaves a singleton last dimension whose stride can be
            # greater than 1 even after contiguous(); view(uint8) rejects it.
            # After this, element [e, offs_bn//2, k_group] is the uint8 byte
            # holding the two int4 ZPs for output channels offs_bn and offs_bn+1.
            if w13_qzeros is not None:
                w13_qzeros = (
                    w13_qzeros.contiguous()
                    .view(torch.uint8)
                    .transpose(1, 2)
                    .contiguous()
                )
            if w2_qzeros is not None:
                w2_qzeros = (
                    w2_qzeros.contiguous()
                    .view(torch.uint8)
                    .transpose(1, 2)
                    .contiguous()
                )
        else:
            # MoeWNA16 uses N-first uint8 weights and scales.
            w13_uint8 = w13.view(torch.uint8)
            w2_uint8 = w2.view(torch.uint8)
        return (
            w13_uint8,
            w2_uint8,
            w13_scale,
            w2_scale,
            None,
            None,
            None,
            None,
            w13_qzeros,
            w2_qzeros,
            None,
            None,
            w13_bias,
            w2_bias,
        )
    else:
        raise ValueError(f"Unsupported wna16 MoE backend: {backend.value}")
