# SPDX-License-Identifier: Apache-2.0
"""Fused MoE kernel."""
import functools
import json
import os
# torch.compile needs typing.List. It will fail torch.library.infer_schema
# otherwise
from typing import List  # noqa: UP035
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.logger import init_logger
# yapf: disable
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig, get_config_quant_dtype)
from vllm.model_executor.layers.fused_moe.cutlass_moe import (
    _valid_cutlass_block_scaled_grouped_gemm,
    run_cutlass_block_scaled_fused_experts)
# yapf: enable
from vllm.model_executor.layers.fused_moe.deep_gemm_moe import (
    _valid_deep_gemm, deep_gemm_moe_fp8)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoEP)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache, moe_kernel_quantize_input, per_token_group_quant_fp8)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    calculate_tile_tokens_dim)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import (
    dequant_mxfp4)
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils import direct_register_custom_op, is_torch_equal_or_newer
from vllm.utils.deep_gemm import is_blackwell_deep_gemm_e8m0_used


logger = init_logger(__name__)


from vllm_metax.model_executor.layers.fused_moe.fused_moe import (get_moe_configs)
from vllm_metax import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.fused_moe import (should_moe_wna16_use_cuda, 
                                                            write_zeros_to_output)


def get_default_config(
    M: int,
    E: int,
    N: int,
    K: int,
    topk: int,
    dtype: Optional[str],
    is_marlin: bool,
    block_shape: Optional[list[int]] = None,
) -> dict[str, int]:
    if dtype == "fp8_w8a8" and block_shape is not None:
        # Block-wise quant: BLOCK_SIZE_N must be divisible by block_shape[0]
        # BLOCK_SIZE_K must be divisible by block_shape[1]
        # num_stages=3 can cause triton.runtime.errors.OutOfResources
        # on ROCm, set it to 2 instead.
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": block_shape[0],
            "BLOCK_SIZE_K": block_shape[1],
            "GROUP_SIZE_M": 32,
            "num_warps": 4,
            "num_stages": 3,
        }
    # ┌------------------------  Metax Modification -------------------------┐
    # TODO: missing config for BLOCK_SIZE_K
    # elif dtype in ["int4_w4a16", "int8_w8a16"] and block_shape is not None:
    #     # moe wna16 kernels
    #     # only set BLOCK_SIZE_M
    #     # BLOCK_SIZE_N and BLOCK_SIZE_K would be set later
    #     bit = 4 if dtype == "int4_w4a16" else 8
    #     use_moe_wna16_cuda = should_moe_wna16_use_cuda(M * topk,
    #                                                    block_shape[1], E, bit)
    #     if use_moe_wna16_cuda:
    #         config = {"BLOCK_SIZE_M": min(16, M)}
    #     elif M <= 20:
    #         config = {"BLOCK_SIZE_M": 16, "GROUP_SIZE_M": 1}
    #     elif M <= 40:
    #         config = {"BLOCK_SIZE_M": 32, "GROUP_SIZE_M": 1}
    #     else:
    #         config = {"BLOCK_SIZE_M": 64, "GROUP_SIZE_M": 1}
    # └------------------------- Metax Modification -------------------------┘
    elif is_marlin:
        for block_size_m in [8, 16, 32, 48, 64]:
            if M * topk / E / block_size_m < 0.9:
                break
        return {"BLOCK_SIZE_M": block_size_m}
    elif M <= E:
        config = {
            "BLOCK_SIZE_M": 16,
            "BLOCK_SIZE_N": 32,
            "BLOCK_SIZE_K": 64,
            "GROUP_SIZE_M": 1,
        }
    else:
        config = {
            "BLOCK_SIZE_M": 64,
            "BLOCK_SIZE_N": 64,
            "BLOCK_SIZE_K": 32,
            "GROUP_SIZE_M": 8,
        }
    return config

def try_get_optimal_moe_config(
    w1_shape: tuple[int, ...],
    w2_shape: tuple[int, ...],
    top_k: int,
    dtype: Optional[str],
    M: int,
    block_shape: Optional[list[int]] = None,
# ┌------------------------  Metax Modification -------------------------┐
    H: int = 0,
# └------------------------- Metax Modification -------------------------┘
):
    from vllm.model_executor.layers.fused_moe import get_config
    override_config = get_config()
    if override_config:
        config = override_config
    else:
        # First try to load optimal config from the file
        E, _, N = w2_shape
# ┌------------------------  Metax Modification -------------------------┐
        # TODO: why we need N * 2
        # if dtype == "int4_w4a16":
        #     N = N * 2
# └------------------------- Metax Modification -------------------------┘
        block_n = block_shape[0] if block_shape else 0
        block_k = block_shape[1] if block_shape else 0
# ┌------------------------  Metax Modification -------------------------┐
        configs = get_moe_configs(E, N, dtype, block_n, block_k, H)
# └------------------------- Metax Modification -------------------------┘

        if configs:
            # If an optimal configuration map has been found, look up the
            # optimal config
            config = configs[min(configs.keys(), key=lambda x: abs(x - M))]
        else:
            # Else use the default config
            config = get_default_config(M, E, N, w1_shape[2], top_k, dtype,
                                        block_shape)
    return config

def get_config_dtype_str(
        dtype: torch.dtype,
        use_int4_w4a16: Optional[bool] = False,
        # ┌------------------------  Metax Modification -------------------------┐
        use_int8_w8a8: Optional[bool] = False,
        # └------------------------- Metax Modification -------------------------┘
        use_int8_w8a16: Optional[bool] = False,
        use_fp8_w8a8: Optional[bool] = False) -> Optional[str]:
    if use_fp8_w8a8:
        return "fp8_w8a8"
    # ┌------------------------  Metax Modification -------------------------┐
    elif use_int8_w8a8:
        return "int8_w8a8"
    # └------------------------- Metax Modification -------------------------┘
    elif use_int8_w8a16:
        return "int8_w8a16"
    elif use_int4_w4a16:
        return "int4_w4a16"
    elif dtype == torch.float:
        # avoiding cases where kernel fails when float32 MoE
        # use fp16/bfloat16 configs
        return "float32"
    return None

@triton.jit
def fused_moe_kernel_gptq_awq(
        # Pointers to matrices
        a_ptr,
        b_ptr,
        c_ptr,
        b_scale_ptr,
        b_zp_ptr,
        topk_weights_ptr,
        sorted_token_ids_ptr,
        expert_ids_ptr,
        num_tokens_post_padded_ptr,
        # Matrix dimensions
        N: tl.constexpr,
        K: tl.constexpr,
        EM,
        num_valid_tokens,
        # The stride variables represent how much to increase the ptr by when
        # moving by 1 element in a particular dimension. E.g. `stride_am` is
        # how much to increase `a_ptr` by to get the element one row down
        # (A has M rows).
        stride_am,
        stride_ak,
        stride_be,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_bse,
        stride_bsk,
        stride_bsn,
        stride_bze,
        stride_bzk,
        stride_bzn,
        block_k_diviable: tl.constexpr,
        group_size: tl.constexpr,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        GROUP_SIZE_M: tl.constexpr,
        # ┌------------------------  Metax Modification -------------------------┐
        SPLIT_K: tl.constexpr,
        ACCF32: tl.constexpr,
        # └------------------------- Metax Modification -------------------------┘
        MUL_ROUTED_WEIGHT: tl.constexpr,
        top_k: tl.constexpr,
        compute_type: tl.constexpr,
        has_zp: tl.constexpr,
        use_int4_w4a16: tl.constexpr,
        use_int8_w8a16: tl.constexpr):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(
        tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

    off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        write_zeros_to_output(c_ptr, stride_cm, stride_cn, pid_n, N,
                              offs_token, token_mask, BLOCK_SIZE_M,
                              BLOCK_SIZE_N, compute_type)
        return

    offs_bn = (pid_n * BLOCK_SIZE_N +
               tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am +
                      offs_k[None, :] * stride_ak)

    if use_int4_w4a16:
        b_ptrs = b_ptr + off_experts * stride_be + \
            (offs_k[:, None] // 2) * stride_bk + offs_bn[None, :] * \
                stride_bn
        b_shifter = (offs_k[:, None] % 2) * 4
    elif use_int8_w8a16:
        b_ptrs = b_ptr + off_experts * stride_be + \
            offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    if not has_zp and use_int4_w4a16:
        b_zp_num = 8
    if not has_zp and use_int8_w8a16:
        b_zp_num = 128
    elif has_zp and use_int4_w4a16:
        b_zp_shifter = (offs_bn[None, :] % 2) * 4

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.

        if not block_k_diviable:
            k_mask = offs_k[:, None] < K - k * BLOCK_SIZE_K
            k_other = 0.0
        else:
            k_mask = None
            k_other = None

        a = tl.load(a_ptrs,
                    mask=token_mask[:, None] &
                    (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0)
        b = tl.load(b_ptrs)
        if use_int4_w4a16:
            b = (b >> b_shifter) & 0xF

        b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + \
            offs_bn[None, :] * stride_bsn + \
            ((offs_k[:, None] + BLOCK_SIZE_K * k) // group_size) * \
                stride_bsk
        b_scale = tl.load(b_scale_ptrs, mask=k_mask, other=k_other)
        b_scale = b_scale.to(tl.float32)

        if has_zp and use_int4_w4a16:
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_zp_ptrs = b_zp_ptr + off_experts * stride_bze + \
                (offs_bn[None, :] // 2) * stride_bzn + \
                offs_k_true * stride_bzk
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
            b_zp = ((b_zp >> b_zp_shifter) & 0xF)
            b_zp = b_zp.to(tl.float32)
        elif has_zp and use_int8_w8a16:
            offs_k_true = (offs_k[:, None] + BLOCK_SIZE_K * k) // group_size
            b_zp_ptrs = b_zp_ptr + off_experts * stride_bze + \
                offs_bn[None, :] * stride_bzn + \
                offs_k_true * stride_bzk
            b_zp = tl.load(b_zp_ptrs, mask=k_mask, other=k_other)
            b_zp = b_zp.to(tl.float32)

        # We accumulate along the K dimension.
        if has_zp:
            b = ((b.to(tl.float32) - b_zp) * b_scale).to(compute_type)
        else:
            b = ((b.to(tl.float32) - b_zp_num) * b_scale).to(compute_type)
        accumulator = tl.dot(a, b, acc=accumulator)

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        if use_int4_w4a16:
            b_ptrs += (BLOCK_SIZE_K // 2) * stride_bk
        else:
            b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token,
                             mask=token_mask,
                             other=0)
        accumulator = accumulator * moe_weight[:, None]

    accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[
        None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

# ┌------------------------  Metax Modification -------------------------┐
@triton.heuristics(
    {
        "UPGRADE": lambda args: math.ceil((args["EM"] * args["N"]) / (args["BLOCK_SIZE_M"] * args["BLOCK_SIZE_N"])).bit_length() > 31,
    }
)
@triton.heuristics(
    {
        "UPGRADE_A_OFFS": lambda args: (args["num_valid_tokens"] // args["top_k"] * args["stride_am"] + args["BLOCK_SIZE_K"] * args["stride_ak"]).bit_length() > 31,
    }
)
@triton.heuristics(
    {
        "UPGRADE_B_OFFS": lambda args: ((args["E"]-1) * args["stride_be"] + (args["N"]-1) * args["stride_bn"]+(args["K"]-1) * args["stride_bk"]).bit_length() > 31,
    }
)
# └------------------------- Metax Modification -------------------------┘
@triton.jit
def fused_moe_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    b_bias_ptr,
    a_scale_ptr,
    b_scale_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    # Matrix dimensions
    # ┌------------------------  Metax Modification -------------------------┐
    E,          # B.shape[0]
    # └------------------------- Metax Modification -------------------------┘
    N,
    K,
    EM,
    num_valid_tokens,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    stride_asm,
    stride_ask,
    stride_bse,
    stride_bsk,
    stride_bsn,
    stride_bbe,  # bias expert stride
    stride_bbn,  # bias N stride
    # Block size for block-wise quantization
    group_n: tl.constexpr,
    group_k: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    # ┌------------------------  Metax Modification -------------------------┐
    SPLIT_K: tl.constexpr,
    ACCF32: tl.constexpr,
    # └------------------------- Metax Modification -------------------------┘
    MUL_ROUTED_WEIGHT: tl.constexpr,
    top_k: tl.constexpr,
    compute_type: tl.constexpr,
    use_fp8_w8a8: tl.constexpr,
    use_int8_w8a8: tl.constexpr,
    use_int8_w8a16: tl.constexpr,
    per_channel_quant: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    # ┌------------------------  Metax Modification -------------------------┐
    UPGRADE: tl.constexpr,
    UPGRADE_A_OFFS: tl.constexpr,
    UPGRADE_B_OFFS: tl.constexpr,
    FAST_F32_TO_BF16: tl.constexpr
    # └------------------------- Metax Modification -------------------------┘
):
    """
    Implements the fused computation for a Mixture of Experts (MOE) using
    token and expert matrices.

    Key Parameters:
    - A: The input tensor representing tokens with shape (*, K), where '*' can
        be any shape representing batches and K is the feature dimension of
        each token.
    - B: The stacked MOE weight tensor with shape (E, N, K), where E is
        the number of experts, K is the input feature dimension, and N is
        the output feature dimension.
    - C: The output cache tensor with shape (M, topk, N), where M is the
        total number of tokens post padding, topk is the number of times
        each token is repeated, and N is the output feature dimension.
    - sorted_token_ids: A tensor containing the sorted indices of tokens,
        repeated topk times and arranged by the expert index they are
        assigned to.
    - expert_ids: A tensor containing the indices of the expert for each
        block. It determines which expert matrix from B should be used for
        each block in A.
    This kernel performs the multiplication of a token by its corresponding
    expert matrix as determined by `expert_ids`. The sorting of
    `sorted_token_ids` by expert index and padding ensures divisibility by
    BLOCK_SIZE_M, which is necessary to maintain consistency in block matrix
    multiplication across different blocks processed by the same expert.
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    
# ┌------------------------  Metax Modification -------------------------┐
    if UPGRADE:
        pid = tl.program_id(axis=0).to(tl.int64)
        pid_z = tl.program_id(axis=1).to(tl.int64)
    else:
        pid = tl.program_id(axis=0)
        pid_z = tl.program_id(axis=1)
# └------------------------- Metax Modification -------------------------┘
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return
    offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(
        tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + offs_token_id)
    token_mask = offs_token < num_valid_tokens

# ┌------------------------  Metax Modification -------------------------┐
    if UPGRADE_B_OFFS:
        off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    else:
        off_experts = tl.load(expert_ids_ptr + pid_m)
        
    if UPGRADE_A_OFFS:
        offs_token = offs_token.to(tl.int64)
# └------------------------- Metax Modification -------------------------┘

    if off_experts == -1:
        # -----------------------------------------------------------
        # Write back zeros to the output when the expert is not
        # in the current expert parallel rank.
        write_zeros_to_output(c_ptr, stride_cm, stride_cn, pid_n, N,
                              offs_token, token_mask, BLOCK_SIZE_M,
                              BLOCK_SIZE_N, compute_type)
        return
# ┌------------------------  Metax Modification -------------------------┐
    if UPGRADE_B_OFFS:
        offs_bn = (pid_n * BLOCK_SIZE_N +
                   tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    else:
        offs_bn = (pid_n * BLOCK_SIZE_N +
                   tl.arange(0, BLOCK_SIZE_N)) % N

        
    # offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_k = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
# └------------------------- Metax Modification -------------------------┘

    a_ptrs = a_ptr + (offs_token[:, None] // top_k * stride_am +
                      offs_k[None, :] * stride_ak)

    b_ptrs = b_ptr + off_experts * stride_be + (offs_k[:, None] * stride_bk +
                                                offs_bn[None, :] * stride_bn)
    if use_int8_w8a16:
        b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bn[
            None, :] * stride_bsn
        b_scale = tl.load(b_scale_ptrs)

# ┌------------------------  Metax Modification -------------------------┐
    if use_int8_w8a8:  
        a_scale = tl.load(a_scale_ptr+(offs_token[:, None] // top_k * stride_asm),mask=token_mask[:, None],other=0.0)
        b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bn[
            None, :] * stride_bsn
        b_scale = tl.load(b_scale_ptrs)
# └------------------------- Metax Modification -------------------------┘

    if use_fp8_w8a8:
        # block-wise
        if group_k > 0 and group_n > 0:
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            offs_bsn = offs_bn // group_n
            b_scale_ptrs = (b_scale_ptr + off_experts * stride_bse +
                            offs_bsn * stride_bsn)
        # channel-wise
        elif per_channel_quant:
            b_scale_ptrs = b_scale_ptr + off_experts * stride_bse + offs_bn[
                None, :] * stride_bsn
            b_scale = tl.load(b_scale_ptrs)
            # Load per-token scale for activations
            a_scale_ptrs = a_scale_ptr + (offs_token // top_k) * stride_asm
            a_scale = tl.load(a_scale_ptrs, mask=token_mask, other=0.0)[:,
                                                                        None]
        # tensor-wise
        else:
            a_scale = tl.load(a_scale_ptr)
            b_scale = tl.load(b_scale_ptr + off_experts)
    if HAS_BIAS:
        # bias shape: [num_experts, N]
        bias_ptrs = b_bias_ptr + off_experts * stride_bbe + offs_bn * stride_bbn
        bias = tl.load(bias_ptrs, mask=(offs_bn < N), other=0.0)
    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.

# ┌------------------------  Metax Modification -------------------------┐
    # accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32 if use_int8_w8a8 else tl.float32)
# └------------------------- Metax Modification -------------------------┘
    
# ┌------------------------  Metax Modification -------------------------┐
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        # Load the next block of A and B, generate a mask by checking the
        # K dimension.
        a = tl.load(a_ptrs,
                    mask=token_mask[:, None] &
                    (offs_k[None, :] < K - k * BLOCK_SIZE_K * SPLIT_K),
                    other=0.0)
        b = tl.load(b_ptrs,
                    mask=offs_k[:, None] < K - k * BLOCK_SIZE_K * SPLIT_K,
                    other=0.0)
        # We accumulate along the K dimension.
        if use_int8_w8a16:
            accumulator = tl.dot(a, b.to(compute_type), acc=accumulator)
        elif use_int8_w8a8:
            a = a.to(tl.int8)
            accumulator += tl.dot(a, b,out_dtype=accumulator.dtype)
        elif use_fp8_w8a8:
            if group_k > 0 and group_n > 0:
                k_start = k * BLOCK_SIZE_K * SPLIT_K
                offs_ks = k_start // group_k
                a_scale = tl.load(a_scale_ptrs + offs_ks * stride_ask,
                                  mask=token_mask,
                                  other=0.0)
                b_scale = tl.load(b_scale_ptrs + offs_ks * stride_bsk)

                accumulator += tl.dot(a, b) * a_scale[:,
                                                      None] * b_scale[None, :]
            else:
                if use_fp8_w8a8:
                    # acc used to enable fp8_fast_accum
                    accumulator = tl.dot(a, b, acc=accumulator)
                else:
                    accumulator += tl.dot(a, b)
        else:
            accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak * SPLIT_K
        b_ptrs += BLOCK_SIZE_K * stride_bk * SPLIT_K
# └------------------------- Metax Modification -------------------------┘
        
    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token,
                             mask=token_mask,
                             other=0)
        accumulator = accumulator * moe_weight[:, None]
    if use_int8_w8a16:
        accumulator = (accumulator * b_scale).to(compute_type)
# ┌------------------------  Metax Modification -------------------------┐
    elif use_int8_w8a8:
        accumulator = accumulator.to(tl.float32)
        accumulator = (accumulator * a_scale * b_scale)
        if not ACCF32:
            if FAST_F32_TO_BF16:
                accumulator = accumulator.to(compute_type, "rtne_no_nan")
            else:
                accumulator = accumulator.to(compute_type)
# └------------------------- Metax Modification -------------------------┘
    elif use_fp8_w8a8:
        if group_k > 0 and group_n > 0:
            accumulator = accumulator.to(compute_type)
        else:
            accumulator = (accumulator * a_scale * b_scale).to(compute_type)
    else:
        accumulator = accumulator.to(compute_type)
    # -----------------------------------------------------------
    # Write back the block of the output
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[
        None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
# ┌------------------------  Metax Modification -------------------------┐
    if SPLIT_K == 1:
        tl.store(c_ptrs, accumulator, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask)
# └------------------------- Metax Modification -------------------------┘

def invoke_fused_moe_kernel(A: torch.Tensor,
                            B: torch.Tensor,
                            C: torch.Tensor,
                            A_scale: Optional[torch.Tensor],
                            B_scale: Optional[torch.Tensor],
                            B_zp: Optional[torch.Tensor],
                            topk_weights: Optional[torch.Tensor],
                            sorted_token_ids: torch.Tensor,
                            expert_ids: torch.Tensor,
                            num_tokens_post_padded: torch.Tensor,
                            mul_routed_weight: bool,
                            top_k: int,
                            config: dict[str, Any],
                            compute_type: tl.dtype,
                            use_fp8_w8a8: bool,
                            use_int8_w8a8: bool,
                            use_int8_w8a16: bool,
                            use_int4_w4a16: bool,
# ┌------------------------  Metax Modification -------------------------┐
                            orig_acc_dtype: torch.dtype,
# └------------------------- Metax Modification -------------------------┘
                            per_channel_quant: bool,
                            block_shape: Optional[list[int]] = None,
                            B_bias: Optional[torch.Tensor] = None) -> None:
    assert topk_weights is not None or not mul_routed_weight
    assert topk_weights is None or topk_weights.stride(1) == 1
    assert sorted_token_ids.stride(0) == 1

    if use_fp8_w8a8 or use_int8_w8a8:
        assert B_scale is not None
        assert (block_shape is None or triton.cdiv(B.shape[-2], block_shape[0])
                == B_scale.shape[-2])
        assert (block_shape is None or triton.cdiv(B.shape[-1], block_shape[1])
                == B_scale.shape[-1])

    elif use_int8_w8a16 or use_int4_w4a16:
        assert B_scale is not None
        assert block_shape is None or block_shape[0] == 0
    else:
        assert A_scale is None
        assert B_scale is None

    M = A.shape[0]
    num_tokens = M * top_k

    EM = sorted_token_ids.shape[0]
    if A.shape[0] < config["BLOCK_SIZE_M"]:
        # optimize for small batch_size.
        # We assume that top_ids of each token is unique, so
        # so num_valid_experts <= batch_size <= BLOCK_SIZE_M,
        # and we can skip some invalid blocks.
        EM = min(sorted_token_ids.shape[0],
                 A.shape[0] * top_k * config['BLOCK_SIZE_M'])
    grid = lambda META: (triton.cdiv(EM, META['BLOCK_SIZE_M']) * triton.cdiv(
# ┌------------------------  Metax Modification -------------------------┐
        B.shape[1], META['BLOCK_SIZE_N']), META['SPLIT_K'])
# └------------------------- Metax Modification -------------------------┘
    HAS_BIAS = B_bias is not None
    if (use_int8_w8a16 or use_int4_w4a16) and \
            block_shape is not None and block_shape[1] > 0:
        assert B_scale is not None and B_scale.ndim == 3
        assert B_zp is None or B_zp.ndim == 3

        use_moe_wna16_cuda = should_moe_wna16_use_cuda(
            num_valid_tokens=num_tokens,
            group_size=block_shape[1],
            num_experts=B.shape[0],
            bit=4 if use_int4_w4a16 else 8)
# ┌-------------------------------  Metax Modification --------------------------------┐
        # TODO: missing config for BLOCK_SIZE_K
        # config = config.copy()
        # config.update(
        #     get_moe_wna16_block_config(config=config,
        #                                use_moe_wna16_cuda=use_moe_wna16_cuda,
        #                                num_valid_tokens=num_tokens,
        #                                size_k=A.shape[1],
        #                                size_n=B.shape[1],
        #                                num_experts=B.shape[1],
        #                                group_size=block_shape[1],
        #                                real_top_k=top_k,
        #                                block_size_m=config["BLOCK_SIZE_M"]))

        if False and use_moe_wna16_cuda:
            bit = 4 if use_int4_w4a16 else 8
            ops.moe_wna16_gemm(A, C, B, B_scale, B_zp,
                               topk_weights if mul_routed_weight else None,
                               sorted_token_ids, expert_ids,
                               num_tokens_post_padded, top_k,
                               config["BLOCK_SIZE_M"], config["BLOCK_SIZE_N"],
                               config["BLOCK_SIZE_K"] * config["SPLIT_K"], bit)
            return
# └-------------------------------- Metax Modification --------------------------------┘
        fused_moe_kernel_gptq_awq[grid](
            A,
            B,
            C,
            B_scale,
            B_zp,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            B.shape[1],
            A.shape[1],
            EM,
            num_tokens,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            C.stride(1),
            C.stride(2),
            B_scale.stride(0),
            B_scale.stride(2),
            B_scale.stride(1),
            B_zp.stride(0) if B_zp is not None else 0,
            B_zp.stride(2) if B_zp is not None else 0,
            B_zp.stride(1) if B_zp is not None else 0,
        # ┌-------------------------------  Metax Modification --------------------------------┐
            block_k_diviable=A.shape[1] % config["BLOCK_SIZE_K"] * config["SPLIT_K"] == 0,
        # └-------------------------------- Metax Modification --------------------------------┘
            group_size=block_shape[1],
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            compute_type=compute_type,
            has_zp=B_zp is not None,
            use_int4_w4a16=use_int4_w4a16,
            use_int8_w8a16=use_int8_w8a16,
            **config,
        )
    else:
        config = config.copy()
        BLOCK_SIZE_K = config.pop("BLOCK_SIZE_K")
        if block_shape is not None:
            BLOCK_SIZE_K = min(BLOCK_SIZE_K, min(block_shape[0],
                                                 block_shape[1]))
        fused_moe_kernel[grid](
            A,
            B,
            C,
            B_bias,
            A_scale,
            B_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
# ┌------------------------  Metax Modification -------------------------┐
            B.shape[0],
# └------------------------- Metax Modification -------------------------┘
            B.shape[1],
            B.shape[2],
            EM,
            num_tokens,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(2),
            B.stride(1),
            C.stride(1),
            C.stride(2),
            A_scale.stride(0)
            if A_scale is not None and A_scale.ndim == 2 else 0,
            A_scale.stride(1)
            if A_scale is not None and A_scale.ndim == 2 else 0,
            B_scale.stride(0)
            if B_scale is not None and B_scale.ndim >= 2 else 0,
            B_scale.stride(2)
            if B_scale is not None and B_scale.ndim == 3 else 0,
            B_scale.stride(1)
            if B_scale is not None and B_scale.ndim >= 2 else 0,
            B_bias.stride(0) if B_bias is not None else 0,
            B_bias.stride(1) if B_bias is not None else 0,
            0 if block_shape is None else block_shape[0],
            0 if block_shape is None else block_shape[1],
            MUL_ROUTED_WEIGHT=mul_routed_weight,
            top_k=top_k,
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            per_channel_quant=per_channel_quant,
            HAS_BIAS=HAS_BIAS,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
# ┌------------------------  Metax Modification -------------------------┐
            FAST_F32_TO_BF16 = True,
# └------------------------- Metax Modification -------------------------┘
            **config,
        )
# ┌------------------------  Metax Modification -------------------------┐
    if config["ACCF32"]:
       C = C.to(orig_acc_dtype)
# └------------------------- Metax Modification -------------------------┘

def fused_experts_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    inplace: bool = False,
    activation: str = "silu",
    is_act_and_mul: bool = True,
    apply_router_weight_on_input: bool = False,
    use_fp8_w8a8: bool = False,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    per_channel_quant: bool = False,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    w1_zp: Optional[torch.Tensor] = None,
    w2_zp: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    block_shape: Optional[list[int]] = None,
    w1_bias: Optional[torch.Tensor] = None,
    w2_bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    # Check constraints.
    if use_int4_w4a16:
        assert hidden_states.shape[1] // 2 == w1.shape[
            2], "Hidden size mismatch"
    else:
        assert hidden_states.shape[1] == w1.shape[2], (
            f"Hidden size mismatch {hidden_states.shape[1]} != {w1.shape[2]}")

    assert topk_weights.shape == topk_ids.shape, "topk shape mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w1.stride(-1) == 1, "Stride of last dimension must be 1"
    assert w2.stride(-1) == 1, "Stride of last dimension must be 1"
    assert hidden_states.dtype in [
        torch.float32, torch.float16, torch.bfloat16
    ]
# ┌------------------------  Metax Modification -------------------------┐
    H = hidden_states.shape[-1]
# └------------------------- Metax Modification -------------------------┘
    num_tokens = hidden_states.shape[0]
    E, N, _ = w1.shape
    K = w2.shape[1]
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.shape[1]
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = envs.VLLM_FUSED_MOE_CHUNK_SIZE
    M = min(num_tokens, CHUNK_SIZE)
    config_dtype = get_config_dtype_str(use_fp8_w8a8=use_fp8_w8a8,
        # ┌------------------------  Metax Modification -------------------------┐
                                        use_int8_w8a8=use_int8_w8a8,
        # └------------------------- Metax Modification -------------------------┘
                                        use_int8_w8a16=use_int8_w8a16,
                                        use_int4_w4a16=use_int4_w4a16,
                                        dtype=hidden_states.dtype)

    qtype = get_config_quant_dtype(use_fp8_w8a8=use_fp8_w8a8,
                                    use_int8_w8a8=use_int8_w8a8,
                                    use_int8_w8a16=use_int8_w8a16,
                                    use_int4_w4a16=use_int4_w4a16)

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        w1.shape,
        w2.shape,
        top_k_num,
        config_dtype,
        block_shape=block_shape,
        # ┌------------------------  Metax Modification -------------------------┐
        H=H,
        # └------------------------- Metax Modification -------------------------┘
    )

    config = get_config_func(M)

    # We can reuse the memory between these because by the time we need
    # cache3, we're done with cache1
    # ┌------------------------  Metax Modification -------------------------┐
    stage1_config = config["stage1"] if "stage1" in config else config
    stage2_config = config["stage2"] if "stage2" in config else config
    
    if 'ACCF32' not in stage1_config:
        stage1_config['ACCF32'] = False
    if 'ACCF32' not in stage2_config:
        stage2_config['ACCF32'] = False
    if 'SPLIT_K' not in stage1_config:
        stage1_config['SPLIT_K'] = 1
    if 'SPLIT_K' not in stage2_config:
        stage2_config['SPLIT_K'] = 1    

    if stage1_config['ACCF32']:
       acc_type1 = torch.float32
    else:
       acc_type1 = hidden_states.dtype
    if stage2_config['ACCF32']:
       acc_type2 = torch.float32
    else:
       acc_type2 = hidden_states.dtype
       

    if stage1_config['SPLIT_K'] > 1:
        intermediate_cache1 = torch.zeros((M, topk_ids.shape[1], N),
                                          device=hidden_states.device,
                                          dtype=acc_type1)
    else:
        intermediate_cache1 = torch.empty((M, topk_ids.shape[1], N),
                                          device=hidden_states.device,
                                          dtype=hidden_states.dtype)
        
    if stage2_config['SPLIT_K'] > 1:
        intermediate_cache3 = torch.zeros((M, topk_ids.shape[1], w2.shape[1]),
                                          device=hidden_states.device,
                                          dtype=acc_type2)
    else:
        intermediate_cache3 = torch.empty((M, topk_ids.shape[1], w2.shape[1]),
                                          device=hidden_states.device,
                                          dtype=hidden_states.dtype)
    # └------------------------- Metax Modification -------------------------┘
        
    # This needs separate memory since it's used concurrently with cache1
    intermediate_cache2 = torch.empty((M * top_k_num, N // 2),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)

    if hidden_states.dtype == torch.bfloat16:
        compute_type = tl.bfloat16
    elif hidden_states.dtype == torch.float16:
        compute_type = tl.float16
    elif hidden_states.dtype == torch.float32:
        compute_type = tl.float32
    else:
        raise ValueError(f"Unsupported compute_type: {hidden_states.dtype}")

    if inplace:
        out_hidden_states = hidden_states
    else:
        out_hidden_states = torch.empty_like(hidden_states)

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (chunk * CHUNK_SIZE,
                                          min((chunk + 1) * CHUNK_SIZE,
                                              num_tokens))
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape

        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust the intermediate cache size and config for the last
            # chunk. Note that in most cases we only have one chunk
            # so the cache size and config are already set correctly and
            # do not need to be adjusted.
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[:tokens_in_chunk *
                                                      topk_ids.shape[1]]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
            config = get_config_func(tokens_in_chunk)

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]
        qcurr_hidden_states, a1q_scale = moe_kernel_quantize_input(
            A=curr_hidden_states,
            A_scale=a1_scale,
            quant_dtype=qtype,
            per_act_token_quant=per_channel_quant,
            block_shape=block_shape)

        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            moe_align_block_size(curr_topk_ids, stage1_config['BLOCK_SIZE_M'],
                                 global_num_experts, expert_map))
        # ┌------------------------  Metax Modification -------------------------┐
        if (stage1_config['BLOCK_SIZE_M'] == 128 and use_int8_w8a8==False and (topk_ids.shape[1] == 1 or topk_ids.shape[1] == 2) and
            (curr_hidden_states.dtype == torch.bfloat16 or curr_hidden_states.dtype == torch.float16) and
            w1.shape[1] % 4 == 0 and w1.shape[2] % 8 == 0):
            ops.fused_moe_kernel(curr_hidden_states, w1, intermediate_cache1,
                                curr_topk_weights, curr_topk_ids, sorted_token_ids,
                                expert_ids, num_tokens_post_padded, False,
                                topk_ids.shape[1], 0)
        else:
            if stage2_config['BLOCK_SIZE_M'] != stage1_config['BLOCK_SIZE_M']:
                sorted_token_ids, expert_ids, num_tokens_post_padded = (
                moe_align_block_size(curr_topk_ids, stage2_config['BLOCK_SIZE_M'], global_num_experts, expert_map))

            qcurr_hidden_states, a1q_scale = moe_kernel_quantize_input(
                A=curr_hidden_states,
                A_scale=a1_scale,
                qtype=qtype,
                per_channel_quant=per_channel_quant,
                block_shape=block_shape)
            
            invoke_fused_moe_kernel(qcurr_hidden_states,
                                w1,
                                intermediate_cache1,
                                a1q_scale,
                                w1_scale,
                                w1_zp,
                                curr_topk_weights,
                                sorted_token_ids,
                                expert_ids,
                                num_tokens_post_padded,
                                apply_router_weight_on_input,
                                top_k_num,
                                stage1_config,
                                compute_type=compute_type,
                                use_fp8_w8a8=use_fp8_w8a8,
                                use_int8_w8a8=use_int8_w8a8,
                                use_int8_w8a16=use_int8_w8a16,
                                use_int4_w4a16=use_int4_w4a16,
                                orig_acc_dtype=hidden_states.dtype,
                                per_channel_quant=per_channel_quant,
                                block_shape=block_shape,
                                B_bias=w1_bias)
        # └------------------------- Metax Modification -------------------------┘

        # Activation function with multiplication
        if activation == "silu" and is_act_and_mul:
            torch.ops._C.silu_and_mul(intermediate_cache2,
                                      intermediate_cache1.view(-1, N))
        elif activation == "gelu" and is_act_and_mul:
            torch.ops._C.gelu_and_mul(intermediate_cache2,
                                      intermediate_cache1.view(-1, N))
        elif activation == "swigluoai" and is_act_and_mul:
            # alpha = 1.702, limit = 7.0
            torch.ops._C.swigluoai_and_mul(intermediate_cache2,
                                           intermediate_cache1.view(-1, N))
        # Activation function without multiplication
        elif activation == "silu":
            intermediate_cache2 = F.silu(intermediate_cache1.view(-1, N))
        elif activation == "gelu":
            intermediate_cache2 = F.gelu(intermediate_cache1.view(-1, N))

        else:
            raise ValueError(f"Unsupported FusedMoe activation: {activation}, "
                             f"with is_act_and_mul={is_act_and_mul}.")


        # ┌------------------------  Metax Modification -------------------------┐
        if (stage2_config['BLOCK_SIZE_M'] == 128 and use_int8_w8a8==False and w2.shape[1] % 4 == 0 and w2.shape[2] % 8 == 0 and
            (hidden_states.dtype == torch.bfloat16 or hidden_states.dtype == torch.float16)):
            ops.fused_moe_kernel(intermediate_cache2, w2, intermediate_cache3,
                                curr_topk_weights, curr_topk_ids, sorted_token_ids,
                                expert_ids, num_tokens_post_padded, True, 1, 0)
        else:
            qintermediate_cache2, a2q_scale = moe_kernel_quantize_input(
                A=intermediate_cache2,
                A_scale=a2_scale,
                qtype=qtype,
                per_channel_quant=per_channel_quant,
                block_shape=block_shape)
            
            if stage2_config['BLOCK_SIZE_M'] != stage1_config['BLOCK_SIZE_M']:
                    sorted_token_ids, expert_ids, num_tokens_post_padded = (
                    moe_align_block_size(curr_topk_ids, stage2_config['BLOCK_SIZE_M'], 
                                         global_num_experts, expert_map))

            invoke_fused_moe_kernel(qintermediate_cache2,
                                w2,
                                intermediate_cache3,
                                a2q_scale,
                                w2_scale,
                                w2_zp,
                                curr_topk_weights,
                                sorted_token_ids,
                                expert_ids,
                                num_tokens_post_padded,
                                not apply_router_weight_on_input,
                                1,
                                stage2_config,
                                compute_type=compute_type,
                                use_fp8_w8a8=use_fp8_w8a8,
                                use_int8_w8a8=use_int8_w8a8,
                                use_int8_w8a16=use_int8_w8a16,
                                use_int4_w4a16=use_int4_w4a16,
                                orig_acc_dtype=hidden_states.dtype,
                                per_channel_quant=per_channel_quant,
                                block_shape=block_shape,
                                B_bias=w2_bias)
        # └------------------------- Metax Modification -------------------------┘
        ops.moe_sum(intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states[begin_chunk_idx:end_chunk_idx])

    return out_hidden_states

# TODO: remove the functions that are unnessessary to export 
import vllm.model_executor.layers.fused_moe
vllm.model_executor.layers.fused_moe.fused_moe.get_moe_configs = get_moe_configs
vllm.model_executor.layers.fused_moe.fused_moe.get_default_config = get_default_config
vllm.model_executor.layers.fused_moe.fused_moe.try_get_optimal_moe_config = try_get_optimal_moe_config
vllm.model_executor.layers.fused_moe.fused_moe.get_config_dtype_str = get_config_dtype_str

vllm.model_executor.layers.fused_moe.fused_moe.fused_moe_kernel_gptq_awq = fused_moe_kernel_gptq_awq
vllm.model_executor.layers.fused_moe.fused_moe.fused_moe_kernel = fused_moe_kernel
vllm.model_executor.layers.fused_moe.fused_moe.invoke_fused_moe_kernel = invoke_fused_moe_kernel
vllm.model_executor.layers.fused_moe.fused_moe.fused_experts_impl = fused_experts_impl










