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

        use_nvfp4 = False
        nvfp4_dispatch = (
            quant_config.quant_dtype == "nvfp4" and envs.VLLM_DEEPEPLL_NVFP4_DISPATCH
        )
        if nvfp4_dispatch:
            use_nvfp4 = True
        qc_a1_gscale_or_scale = (
            quant_config.a1_gscale if nvfp4_dispatch else quant_config.a1_scale
        )
        has_per_token_scales = (
            qc_a1_gscale_or_scale.numel() != 1
            if qc_a1_gscale_or_scale is not None
            else (
                quant_config.a2_scale.numel() != 1
                if quant_config.a2_scale is not None
                else False
            )
        )
        if not use_nvfp4:
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
        (
            expert_x,
            expert_num_tokens,
            handle,
            _,
            hook,
        ) = self.buffer.low_latency_dispatch(
            a1,
            dispatch_topk_ids,
            self.max_tokens_per_rank,
            num_experts,
            use_fp8=self.use_fp8_dispatch,
            **(
                dict(topk_weights=topk_weights)
                if check_deepep_package() == "mxmesh"
                else dict()
            ),
            # /---------------- MetaX Modification ---------------\
            # round_scale=self.use_ue8m0_dispatch,
            # use_ue8m0=self.use_ue8m0_dispatch,
            # **(dict(use_nvfp4=True) if use_nvfp4 else dict()),
            # **(
            #     dict(x_global_scale=qc_a1_gscale_or_scale)
            #     if qc_a1_gscale_or_scale is not None and nvfp4_dispatch
            #     else dict()
            # ),
            # \---------------- MetaX Modification ---------------/
            async_finish=False,
            return_recv_hook=True,
        )
        self.handles[a2a_idx] = handle

        return (
            hook,
            lambda: self._receiver(
                expert_x,
                expert_num_tokens,
                quant_config.a1_scale,
                a1.dtype,
                quant_config,
            ),
        )

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
