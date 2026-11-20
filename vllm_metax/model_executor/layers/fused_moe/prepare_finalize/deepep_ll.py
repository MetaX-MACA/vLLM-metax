# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ll import (
    DeepEPLLPrepareAndFinalize,
)
from vllm.v1.worker.ubatching import dbo_current_ubatch_id
from vllm.v1.worker.ubatching import (
    dbo_enabled,
    dbo_maybe_run_recv_hook,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceDelegate,
)
from importlib import metadata

from vllm_metax.model_executor.layers.fused_moe import (
    masked_int8_moe,
    source_packed_dispatch,
)
from vllm_metax.model_executor.layers.fused_moe.deepep_fixed_geometry import (
    copy_valid_rows,
    map_expert_ids,
    wire_experts,
)

logger = init_logger(__name__)

# DeepEP kernels quantize dispatch inputs in 128 element chunks.
DEEPEP_QUANT_BLOCK_SIZE = 128
DEEPEP_QUANT_BLOCK_SHAPE = [DEEPEP_QUANT_BLOCK_SIZE, DEEPEP_QUANT_BLOCK_SIZE]


def check_deepep_package() -> str:
    """Return the installed DeepEP-compatible backend package."""
    try:
        metadata.distribution("mxmesh")
        return "mxmesh"
    except metadata.PackageNotFoundError:
        try:
            metadata.distribution("deep_ep")
            return "deep_ep"
        except metadata.PackageNotFoundError as exc:
            raise FileNotFoundError(
                "DeepEP low-latency kernels require either the mxmesh "
                "or deep_ep package"
            ) from exc


class MacaDeepEPLLPrepareAndFinalize(DeepEPLLPrepareAndFinalize):
    """Prepare/Finalize using DeepEP low-latency kernels."""

    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        defer_input_quant: bool = False,
    ) -> tuple[Callable, mk.ReceiverType]:
        if defer_input_quant:
            raise NotImplementedError(
                f"{self.__class__.__name__} does not support defer_input_quant=True. "
                "Please select an MoE kernel that accepts quantized inputs."
            )

        # /---------------- MetaX Modification ---------------\
        source_packed = source_packed_dispatch.transport_for(a1)
        # \---------------- MetaX Modification ---------------/
        hidden_size = a1.size(1)
        assert hidden_size in self.SUPPORTED_HIDDEN_SIZES, (
            f"Hidden Size {hidden_size} not in supported list of hidden sizes"
            f"{self.SUPPORTED_HIDDEN_SIZES}"
        )

        a2a_idx = dbo_current_ubatch_id()

        if self.use_fp8_dispatch:
            assert hidden_size % 128 == 0, (
                "DeepEP kernels quantize the inputs in blocks of shape 128"
            )

        nvfp4_dispatch = (
            quant_config.quant_dtype == "nvfp4" and envs.VLLM_DEEPEPLL_NVFP4_DISPATCH
        )
        # NOTE(MetaX): installed DeepEP accepts use_fp8 but has no NVFP4 or
        # packed-UE8M0 dispatch API. Do not silently omit these requested formats.
        # Revalidate when the MACA Buffer API gains the corresponding arguments.
        if nvfp4_dispatch or self.use_ue8m0_dispatch:
            raise NotImplementedError(
                "MetaX DeepEP does not support NVFP4 or packed UE8M0 dispatch"
            )
        has_per_token_scales = (
            quant_config.a1_scale.numel() != 1
            if quant_config.a1_scale is not None
            else (
                quant_config.a2_scale.numel() != 1
                if quant_config.a2_scale is not None
                else False
            )
        )
        assert not has_per_token_scales, (
            "low_latency kernels doesn't support dispatching per-token scales"
        )

        if apply_router_weight_on_input:
            topk = topk_ids.size(1)
            # TODO: this only works for topK=1, will need to update for topK>1
            assert topk == 1, (
                "apply_router_weight_on_input is only implemented for topk=1"
            )
            a1 = a1 * topk_weights.to(a1.dtype)

        # Dispatch
        dispatch_topk_ids = self._map_global_to_physical_ids(topk_ids)
        transport_experts = wire_experts(self.buffer, num_experts)
        if transport_experts != num_experts:
            if not quant_config.use_int4_w4a8 or self.use_fp8_dispatch:
                raise NotImplementedError(
                    "Fixed expert geometry currently supports BF16 W4A8 transport"
                )
            dispatch_topk_ids = map_expert_ids(
                dispatch_topk_ids,
                num_experts,
                transport_experts,
                self.buffer.group_size,
            )
        (
            expert_x,
            expert_num_tokens,
            handle,
            _,
            hook,
        ) = self.buffer.low_latency_dispatch(
            # /---------------- MetaX Modification ---------------\
            a1 if source_packed is None else source_packed,
            # \---------------- MetaX Modification ---------------/
            dispatch_topk_ids,
            self.max_tokens_per_rank,
            transport_experts,
            use_fp8=self.use_fp8_dispatch,
            **(
                dict(topk_weights=topk_weights)
                if check_deepep_package() == "mxmesh"
                else dict()
            ),
            async_finish=False,
            return_recv_hook=True,
        )
        # /---------------- MetaX Modification ---------------\
        if source_packed is not None:
            # Dispatch uses a byte carrier; combine sends the real BF16 hidden size.
            # Preserve source info, expert layout and buffer capacity exactly.
            handle = (handle[0], handle[1], handle[2], hidden_size, handle[4])
        # \---------------- MetaX Modification ---------------/
        self.handles[a2a_idx] = handle
        if not hasattr(self, "_metax_wire_states"):
            self._metax_wire_states: dict[
                int, tuple[int, int, torch.Tensor, torch.Tensor] | None
            ] = {}
        self._metax_wire_states[a2a_idx] = None
        if transport_experts != num_experts:
            assert (
                isinstance(expert_x, torch.Tensor) and expert_x.dtype == torch.bfloat16
            )
            local_experts = num_experts // self.buffer.group_size
            valid_counts = expert_num_tokens[:local_experts]
            combine_view = expert_x
            if source_packed is not None:
                combine_view = source_packed_dispatch.combine_workspace(
                    self.buffer, expert_x, hidden_size, a2a_idx
                )
            self._metax_wire_states[a2a_idx] = (
                num_experts,
                transport_experts,
                combine_view,
                valid_counts,
            )
            expert_x = expert_x[:local_experts]
            expert_num_tokens = valid_counts

        return (
            hook,
            lambda: self._receiver(
                expert_x,
                expert_num_tokens,
                quant_config.a1_scale,
                a1.dtype,
                quant_config,
                # /---------------- MetaX Modification ---------------\
                packed_hidden_size=hidden_size if source_packed is not None else None,
                # \---------------- MetaX Modification ---------------/
            ),
        )

    def _receiver(
        self,
        expert_x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        expert_num_tokens: torch.Tensor,
        a1_scale: torch.Tensor | None,
        a1_dtype: torch.dtype,
        quant_config: FusedMoEQuantConfig,
        # /---------------- MetaX Modification ---------------\
        packed_hidden_size: int | None = None,
        # \---------------- MetaX Modification ---------------/
    ) -> mk.PrepareResultType:
        # /---------------- MetaX Modification ---------------\
        packed_input = None
        if packed_hidden_size is not None:
            if (
                not isinstance(expert_x, torch.Tensor)
                or expert_x.dtype != torch.bfloat16
            ):
                raise ValueError("Packed receiver requires a BF16 byte carrier")
            e, t, width = expert_x.shape
            if (
                width != source_packed_dispatch.carrier_width(packed_hidden_size)
                or not expert_x.is_contiguous()
            ):
                raise ValueError("Unsupported packed receive layout")
            # Pass the ENTIRE padded carrier to native GEMM, as SGL does.
            # No quantization, unpack, compact copy, cat or zero kernel here.
            packed_input = expert_x.view(torch.int8)
            # Shape-only ABI placeholders for the existing modular allocator.
            # Neither is initialized/read by the guarded packed experts path.
            # Keep K=hidden size so output/workspace/combine dimensions stay correct.
            _cache = getattr(self.buffer, "_metax_w4a8_expert_x_scratch", None)
            if _cache is None:
                _cache = self.buffer._metax_w4a8_expert_x_scratch = {}
            _key = ((e, t, packed_hidden_size), expert_x.device)
            if _key not in _cache:
                _cache[_key] = torch.empty(
                    (e, t, packed_hidden_size), device=expert_x.device, dtype=torch.int8
                )
            expert_x = _cache[_key]
            expert_x_scale = torch.empty(
                (e, t, 1), device=expert_x.device, dtype=torch.float32
            )
        elif (
            masked_int8_moe.enabled()
            and (quant_config.use_int8_w8a8 or quant_config.use_int4_w4a8)
            and quant_config.per_act_token_quant
            and quant_config.a1_scale is None
            and quant_config.block_shape is None
            and not self.use_fp8_dispatch
            and isinstance(expert_x, torch.Tensor)
            and expert_x.dtype == torch.bfloat16
        ):
            expert_x, expert_x_scale, packed_input = (
                masked_int8_moe.quantize_and_pack_valid_rows(
                    expert_x, expert_num_tokens
                )
            )
        else:
            expert_x, expert_x_scale = self._do_quant(expert_x, a1_dtype, quant_config)

        expert_tokens_meta = masked_int8_moe.PackedInt8TokensMetadata(
            expert_num_tokens, packed_input
        )
        # \---------------- MetaX Modification ---------------/

        return expert_x, expert_x_scale, expert_tokens_meta, None, None

    def _finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        weight_and_reduce_impl: mk.TopKWeightAndReduce,
        do_async: bool,
    ) -> tuple[Callable, Callable]:
        assert isinstance(weight_and_reduce_impl, TopKWeightAndReduceDelegate), (
            "Weight application and reduction happens in the combine kernel."
        )

        a2a_idx = dbo_current_ubatch_id()
        do_recv_hook = dbo_enabled() or do_async
        handle = self.handles[a2a_idx]
        assert handle is not None

        combine_topk_weights = topk_weights
        if apply_router_weight_on_input:
            # weights have already been applied.
            combine_topk_weights = torch.ones_like(topk_weights)

        combine_topk_ids = self._map_global_to_physical_ids(topk_ids)
        state = getattr(self, "_metax_wire_states", {}).get(a2a_idx)
        if state is not None:
            actual, wire, wire_recv, valid_counts = state
            combine_topk_ids = map_expert_ids(
                combine_topk_ids, actual, wire, self.buffer.group_size
            )
            # Only computed rows enter the native wire view; unused expert
            # slots have zero counts and are never read by combine.
            copy_valid_rows(wire_recv, fused_expert_output, valid_counts)
            fused_expert_output = wire_recv
        # TODO (varun) : Enable zero copy mode
        dbo_maybe_run_recv_hook()
        combined, _, recv_hook = self.buffer.low_latency_combine(
            fused_expert_output,
            combine_topk_ids,
            combine_topk_weights,
            handle,
            async_finish=False,
            zero_copy=False,
            return_recv_hook=do_recv_hook,
            **(dict(out=output) if check_deepep_package() == "deep_ep" else dict()),
        )

        # /------------------------ MetaX Modification -------------------------\
        # MxMesh returns a separate tensor and does not update vLLM's output.
        if check_deepep_package() == "mxmesh":
            if do_recv_hook:
                return recv_hook, lambda: output.copy_(combined)

            output.copy_(combined)
        # \------------------------ MetaX Modification -------------------------/
        return recv_hook, lambda: None
