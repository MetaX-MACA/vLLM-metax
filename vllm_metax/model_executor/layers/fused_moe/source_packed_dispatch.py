# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Opt-in per-token INT8 source-packed DeepEP LL path (model independent).

The source rank quantizes + packs (INT8 values and an FP32 scale per token) and
dispatches that carrier instead of BF16, so the receiver feeds the padded buffer
straight to the masked GEMM.

The scope below exists ONLY inside the opaque MoE custom op: tensors are
explicit op inputs, never hidden compile-time globals. The scope is a plain
module global (not a ContextVar: dynamo cannot trace ContextVar.get, and the
runner's layer-name lookup is traced for layers that are not packed); reset
in finally prevents reuse by the next layer.
DBO, LoRA and routed input/output transforms are excluded. EPLB only
remaps expert ids before dispatch, so it is compatible. Shared
experts are supported: the runner feeds them the original BF16 hidden states.
"""

from collections.abc import Callable

import torch

from vllm.utils.torch_utils import direct_register_custom_op

# Set by the runtime adapter: MoERunner.forward before it is wrapped.
original_runner_forward: Callable[..., torch.Tensor] | None = None

# (runner, layer_name, original hidden, packed carrier) while a packed MoE op runs.
_scope = None


def carrier_width(hidden_size):
    """BF16 elements of one row: hidden_size INT8 values + FP32 scale, 512B aligned.

    SGL layout: 1792 for H=3072, 3328 for H=6144.
    """
    return -(-(hidden_size + 4) // 512) * 256


def transport_for(x):
    if _scope is None:
        return None
    _, _, original, packed = _scope
    # The normal runner may chunk the token dimension. Preserve its exact
    # slicing, including zero-token ranks, without copying payload bytes.
    if x.untyped_storage().data_ptr() != original.untyped_storage().data_ptr():
        raise RuntimeError(
            "Packed MoE runner copied/padded hidden states; unsupported path"
        )
    offset = x.storage_offset() - original.storage_offset()
    if offset < 0 or offset % x.shape[1]:
        raise RuntimeError("Packed MoE input does not align with source token rows")
    start = offset // x.shape[1]
    if start + x.shape[0] > original.shape[0]:
        raise RuntimeError("Packed MoE chunk exceeds source token rows")
    return packed.narrow(0, start, x.shape[0])


def _norm_pack(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from mcoplib.custom_ops import (
        fused_add_rms_norm_dynamic_per_token_quant_padding_output,
    )

    if hidden.dtype != torch.bfloat16 or residual.dtype != torch.bfloat16:
        raise ValueError("Source-packed MoE requires BF16 hidden and residual")
    if hidden.shape != residual.shape or hidden.ndim != 2:
        raise ValueError("Source-packed MoE requires equal 2D hidden/residual shapes")
    if (
        not hidden.is_contiguous()
        or not residual.is_contiguous()
        or not weight.is_contiguous()
    ):
        raise ValueError("Source-packed MoE requires contiguous norm inputs")
    if hidden.shape[0] == 0:
        return hidden.new_empty((0, width)), torch.empty_like(hidden)
    # Native op updates residual IN PLACE, exactly like post_attention_layernorm.
    # Return only fresh outputs; do not return an alias of the mutated residual.
    packed, _, normed, _, _ = fused_add_rms_norm_dynamic_per_token_quant_padding_output(
        hidden, residual, weight, width, epsilon
    )
    return packed, normed


def _norm_pack_nores(
    hidden: torch.Tensor, weight: torch.Tensor, epsilon: float, width: int
) -> tuple[torch.Tensor, torch.Tensor]:  # pdperf_dsv4
    from mcoplib.custom_ops import (
        fused_add_rms_norm_dynamic_per_token_quant_padding_output,
    )

    if hidden.dtype != torch.bfloat16 or hidden.ndim != 2:
        raise ValueError("Source-packed MoE requires BF16 2D norm input")
    if hidden.shape[0] == 0:
        return hidden.new_empty((0, width)), torch.empty_like(hidden)
    packed, _, normed, _, _ = fused_add_rms_norm_dynamic_per_token_quant_padding_output(
        hidden.contiguous(), None, weight, width, epsilon
    )
    return packed, normed


def _norm_pack_nores_fake(
    hidden: torch.Tensor, weight: torch.Tensor, epsilon: float, width: int
) -> tuple[torch.Tensor, torch.Tensor]:
    return hidden.new_empty((hidden.shape[0], width)), torch.empty_like(hidden)


direct_register_custom_op(
    op_name="metax_norm_pack_nores",
    op_func=_norm_pack_nores,
    mutates_args=[],
    fake_impl=_norm_pack_nores_fake,
)


def _norm_pack_fake(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return hidden.new_empty((hidden.shape[0], width)), torch.empty_like(hidden)


direct_register_custom_op(
    op_name="metax_norm_pack",
    op_func=_norm_pack,
    mutates_args=["residual"],
    fake_impl=_norm_pack_fake,
)


def _require_runner(runner):
    kernel = runner._quant_method.moe_kernel
    # These config guards are immutable for this kernel instance. Avoid
    # repeating import/type/config checks for every layer on every eager step.
    # Replacing the quant method/kernel invalidates the identity cache.
    if (
        kernel is not None
        and getattr(runner, "_metax_source_packed_checked_kernel", None) is kernel
    ):
        return
    from vllm.v1.worker.ubatching import dbo_enabled
    from vllm_metax.model_executor.layers.fused_moe.experts.batched_deep_gemm_moe import (
        BatchedDeepGemmExperts,
    )
    from vllm_metax.model_executor.layers.fused_moe.prepare_finalize.deepep_ll import (
        MacaDeepEPLLPrepareAndFinalize,
        check_deepep_package,
    )

    if dbo_enabled():
        raise ValueError("Source-packed MoE does not support DBO")
    pc = runner.moe_config.moe_parallel_config
    if pc.tp_size != 1 or pc.sp_size != 1 or pc.pcp_size != 1 or not pc.use_ep:
        raise ValueError(
            "Source-packed MoE requires TP1, EP, no sequence/context parallelism"
        )
    if (
        runner.routed_input_transform is not None
        or runner.routed_output_transform is not None
    ):
        raise ValueError("Source-packed MoE excludes routed input/output transforms")
    if kernel is None or not isinstance(
        kernel.prepare_finalize, MacaDeepEPLLPrepareAndFinalize
    ):
        raise ValueError(
            "Source-packed MoE requires the MACA DeepEP low-latency modular kernel"
        )
    # Only DeepEP's five-element handle has the byte-carrier layout used below.
    # MxMesh remains supported by the ordinary, unpacked prepare/finalize path.
    if check_deepep_package() != "deep_ep":
        raise ValueError(
            "Source-packed MoE requires deep_ep; MxMesh packed dispatch is unsupported"
        )
    expert = kernel.fused_experts
    if not isinstance(expert, BatchedDeepGemmExperts):
        raise ValueError("Source-packed MoE requires BatchedDeepGemmExperts")
    qc = expert.quant_config
    from vllm_metax import envs as mx_envs

    w4a8 = qc.use_int4_w4a8
    if w4a8:
        from vllm.config import get_current_vllm_config_or_none

        config = get_current_vllm_config_or_none()
        if (
            not mx_envs.VLLM_METAX_MASKED_INT8_MOE
            or not mx_envs.VLLM_METAX_SOURCE_PACKED_MOE
        ):
            raise ValueError(
                "Packed transport requires masked INT8 and source packing switches"
            )
        if config is not None:
            if config.lora_config is not None:
                raise ValueError("W4A8 packed transport excludes LoRA")
            if (
                config.scheduler_config.max_num_batched_tokens
                > kernel.prepare_finalize.max_tokens_per_rank
            ):
                raise ValueError(
                    "W4A8 packed transport requires scheduler chunk <= DeepEP LL buffer capacity"
                )
    if (
        not (qc.use_int8_w8a8 or w4a8)
        or not qc.per_act_token_quant
        or qc.a1_scale is not None
        or qc.a2_scale is not None
        or qc.block_shape is not None
        or kernel.prepare_finalize.use_fp8_dispatch
    ):
        raise ValueError(
            "Source-packed MoE requires dynamic per-token INT8, BF16 dispatch carrier"
        )
    from vllm_metax.model_executor.layers.fused_moe import masked_int8_moe

    act = expert.activation_config
    if (
        not w4a8
        and masked_int8_moe.activation_route(
            runner.moe_config.activation, act.clamp_limit, act.alpha, act.beta
        )
        is None
    ) or runner.routed_experts.apply_router_weight_on_input:
        raise ValueError(
            "Source-packed MoE requires a supported SwiGLU variant and output-side router weights"
        )
    runner._metax_source_packed_checked_kernel = kernel
    from vllm.logger import init_logger

    init_logger(__name__).info_once(
        "Using W4A8 source quant/pack after original MHC/Norm; padded INT8 transport"
        if w4a8
        else "Using source-packed INT8 MoE: fused norm/quant/pack and padded GEMM input"
    )


def _packed_moe(
    hidden: torch.Tensor,
    packed: torch.Tensor,
    router_logits: torch.Tensor,
    input_ids: torch.Tensor | None,
    shared_experts_input: torch.Tensor | None,
    layer_name: str,
) -> torch.Tensor:
    from vllm.model_executor.layers.fused_moe.runner.moe_runner import (
        get_layer_from_name,
    )

    runner_forward = original_runner_forward
    if runner_forward is None:
        raise RuntimeError("Source-packed MoE adapter has not been initialized")

    runner = get_layer_from_name(layer_name)
    _require_runner(runner)
    global _scope
    if _scope is not None:
        raise RuntimeError("Nested source-packed MoE call is unsupported")
    _scope = (runner, layer_name, hidden, packed)
    try:
        # Keep original padding, routing, expert map, modular kernel,
        # reduction and output handling. No direct GEMM or router bypass.
        # The outer custom op hides the inner moe_forward from compilation
        # passes that collect all_moe_layers. Resolve this runner explicitly;
        # do not consume that pass's positional layer-name counter.
        return runner_forward(
            runner, hidden, router_logits, input_ids, shared_experts_input
        )
    finally:
        _scope = None


def _packed_moe_fake(
    hidden: torch.Tensor,
    packed: torch.Tensor,
    router_logits: torch.Tensor,
    input_ids: torch.Tensor | None,
    shared_experts_input: torch.Tensor | None,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(hidden)


direct_register_custom_op(
    op_name="metax_packed_moe",
    op_func=_packed_moe,
    mutates_args=[],
    fake_impl=_packed_moe_fake,
)


def combine_workspace(buffer, received, hidden_size, slot):
    # A packed receive row is narrower than the BF16 combine row. Never reuse
    # that storage as a full-width BF16 tensor. Share scratch across layers on
    # the same DeepEP buffer; DBO is excluded by the source adapter guards.
    cache = getattr(buffer, "_metax_w4a8_combine_scratch", None)
    if cache is None:
        cache = buffer._metax_w4a8_combine_scratch = {}
    shape = (*received.shape[:2], hidden_size)
    key = (slot, shape, received.device)
    if key not in cache:
        cache[key] = received.new_empty(shape)
    return cache[key]
