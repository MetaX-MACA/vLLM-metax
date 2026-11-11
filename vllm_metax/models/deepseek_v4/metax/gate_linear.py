# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
from torch.nn.parameter import Parameter

import vllm._custom_ops as ops
import vllm.envs as envs
from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.utils.torch_utils import direct_register_custom_op


def _maca_fp32_router_gemm_dispatch(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    # Keep the runtime shape split inside a custom op so graph capture does not
    # freeze the token-count branch.
    max_tokens = 48 if weight.shape == (256, 5120) else GateLinear.FP32_MAX_TOKENS
    if x.shape[0] <= max_tokens:
        return ops.fp32_router_gemm(x, weight)
    if weight.dtype == torch.bfloat16:
        return torch.mm(x, weight.T, out_dtype=torch.float32)
    return torch.nn.functional.linear(x.float(), weight)


def _maca_fp32_router_gemm_dispatch_fake(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.float32)


direct_register_custom_op(
    op_name="mx_maca_fp32_router_gemm_dispatch",
    op_func=_maca_fp32_router_gemm_dispatch,
    mutates_args=[],
    fake_impl=_maca_fp32_router_gemm_dispatch_fake,
)


class MacaGateLinear(GateLinear):
    """MoE gate linear layer with three-tier GEMM dispatch:

    1. MACA fp32 router kernel (batch<=32, supported dims)
    2. MACA bf16×bf16→fp32 (bf16 weight + fp32 out_dtype)
    3. F.linear via ReplicatedLinear (ultimate fallback)

    The ``out_dtype`` attribute is mutable and can be set after init
    (e.g. when the required dtype depends on the expert quantization
    method which is only known later).
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        out_dtype: torch.dtype | None = None,
        params_dtype: torch.dtype | None = None,
        force_fp32_compute: bool = False,
        skip_bias_add: bool = False,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        return_bias: bool = True,
    ):
        # NOTE(metax): Keep the full upstream lifecycle, including quantized
        # gates, output/bias settings and state consumed by inherited methods.
        super().__init__(
            input_size,
            output_size,
            bias=bias,
            out_dtype=out_dtype,
            params_dtype=params_dtype,
            force_fp32_compute=force_fp32_compute,
            skip_bias_add=skip_bias_add,
            quant_config=quant_config,
            prefix=prefix,
            return_bias=return_bias,
        )

        # NOTE(metax): MACA supplies the fp32 router kernel and torch.mm
        # out_dtype epilogue. CUDA CuteDSL and ROCm tiers stay disabled by
        # the upstream OOT platform checks, including in set_out_dtype().
        self.allow_fp32_router_gemm = (
            self.is_unquantized
            and not envs.VLLM_BATCH_INVARIANT
            and not bias
            and self.weight.dtype == torch.float32
            and (input_size, output_size) in self.FP32_SUPPORTED_SHAPES
        )
        self._router_gemm_cublas_capable = self._router_gemm_no_bias
        self.allow_cublas_router_gemm = (
            self._router_gemm_cublas_capable
            and not envs.VLLM_BATCH_INVARIANT
            and self.weight.dtype == torch.bfloat16
            and self.out_dtype == torch.float32
        )

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, Parameter | None]:
        # Tier 1: MACA FP32 router GEMM, including the 48-token 5120-wide
        # target shape.
        if self.allow_fp32_router_gemm and x.dtype in (
            torch.float32,
            torch.bfloat16,
        ):
            output = torch.ops.vllm.mx_maca_fp32_router_gemm_dispatch(x, self.weight)
            return self._return(output)

        # Tier 2: MACA bf16 x bf16 -> fp32 epilogue.
        if self.allow_cublas_router_gemm and x.dtype == torch.bfloat16:
            output = torch.mm(x, self.weight.T, out_dtype=torch.float32)
            return self._return(output)

        # Tier 3: complete upstream fallback, including quantized gates.
        return super().forward(x)
