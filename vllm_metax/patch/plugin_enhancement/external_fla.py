# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

import inspect
import os

import torch

if os.getenv("MACA_VLLM_USE_EXTERNAL_FLA", "0") == "1":
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule as external_chunk_gated_delta_rule,
    )
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        ChunkGatedDeltaRule,
    )

    _fla_parameters = inspect.signature(external_chunk_gated_delta_rule).parameters

    def forward_external_fla(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
        core_attn_out: torch.Tensor | None = None,
    ):
        # vLLM recurrent state: [N, HV, V, K]
        fla_state = initial_state
        kwargs = {}

        if "state_v_first" in _fla_parameters:
            kwargs["state_v_first"] = True
        elif "transpose_state_layout" in _fla_parameters:
            kwargs["transpose_state_layout"] = True
        else:
            fla_state = initial_state.transpose(-1, -2).contiguous()

        output, final_state = external_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=fla_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            **kwargs,
        )

        if (
            output_final_state
            and final_state is not None
            and "state_v_first" not in kwargs
            and "transpose_state_layout" not in kwargs
        ):
            final_state = final_state.transpose(-1, -2).contiguous()

        if core_attn_out is not None:
            src = output.squeeze(0).reshape(-1)
            dst = core_attn_out.reshape(-1)
            dst[: src.numel()].copy_(src)

        return output, final_state

    ChunkGatedDeltaRule.forward_native = forward_external_fla
