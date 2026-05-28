# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.model_executor.layers.fused_moe.router.gate_linear import GateLinear


@GateLinear.register_oot
class MacaGateLinear(GateLinear):
    """MoE gate linear layer with three-tier GEMM dispatch:

    1. DSV3 specialized kernel (SM90+, batch<=16, supported dims)
    2. cuBLAS bf16×bf16→fp32 (SM90+ + bf16 + fp32 out_dtype)
    3. F.linear via ReplicatedLinear (ultimate fallback)

    The ``out_dtype`` attribute is mutable and can be set after init
    (e.g. when the required dtype depends on the expert quantization
    method which is only known later).
    """

    # Dimensions supported by the DSV3 specialized kernel
    DSV3_SUPPORTED_NUM_EXPERTS = [256, 384]
    DSV3_SUPPORTED_HIDDEN_SIZES = [7168]

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        out_dtype: torch.dtype | None = None,
        params_dtype: torch.dtype | None = None,
        force_fp32_compute: bool = False,
        prefix: str = "",
    ):
        # is_hopper_or_blackwell = current_platform.is_device_capability(
        #     (9, 0)
        # ) or current_platform.is_device_capability_family(100)
        # can_use_specialized_kernels = (
        #     current_platform.is_cuda() and is_hopper_or_blackwell and not bias
        # )
        can_use_specialized_kernels = not bias

        # If fp32 compute is required and no specialized kernel is available,
        # store weights in fp32 so Tier 3 computes in fp32 natively.
        if force_fp32_compute and not can_use_specialized_kernels:
            params_dtype = torch.float32

        super(GateLinear, self).__init__(
            input_size,
            output_size,
            bias=bias,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=prefix,
        )
        self.out_dtype = out_dtype

        # DSV3 specialized kernel eligibility (SM90+, exact dims)
        self.allow_specialized_router_gemm = can_use_specialized_kernels
        self.allow_dsv3_router_gemm = False

        # cuBLAS bf16→fp32 eligibility
        self.allow_cublas_router_gemm = (
            self.allow_specialized_router_gemm
            and self.weight.dtype == torch.bfloat16
            and self.out_dtype == torch.float32
        )
