# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.


# ---------------------------------------------------
# Note: enable with MACA_VLLM_ENABLE_MCTLASS_PYTHON_API=1
# ---------------------------------------------------

from typing import Any

import torch
import contextlib
from vllm.utils.torch_utils import direct_register_custom_op, is_torch_equal_or_newer

# support W4A8 Per-Channel start
# Init FusedMoeGEMM instance
mctlass_moe_gemm = None
mctlass_scaled_gemm = None
with contextlib.suppress(ImportError):
    if mctlass_moe_gemm is None:
        from mctlassEx import FusedMoeGEMM

        mctlass_moe_gemm = FusedMoeGEMM()

    if mctlass_scaled_gemm is None:
        from mctlassEx import ScaledGEMM

        mctlass_scaled_gemm = ScaledGEMM()


# GEMM
def mctlassEx_fused_moe_w4a8_gemm_per_channel(
    batch_size: int,
    N: int,
    K: int,
    num_experts: int,
    EM: int,
    topk: int,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    b_bias: torch.Tensor,
    topk_weights: torch.Tensor,
    token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
) -> torch.Tensor:
    assert mctlass_moe_gemm is not None, "mctlassMoeGEMM is not imported correctly"
    mctlass_moe_gemm(
        batch_size,
        N,
        K,
        num_experts,
        EM,
        topk,
        a,
        b,
        c,
        a_scales,
        b_scales,
        b_bias,
        topk_weights,
        token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight,
    )
    return c


# Fake
def mctlassEx_fused_moe_w4a8_gemm_per_channel_fake(
    batch_size,
    N,
    K,
    num_experts,
    EM,
    topk,
    a,
    b,
    c,
    a_scales,
    b_scales,
    b_bias,
    topk_weights,
    token_ids,
    expert_ids,
    num_tokens_post_padded,
    mul_routed_weight,
) -> torch.Tensor:
    return c


direct_register_custom_op(
    op_name="mctlassEx_fused_moe_w4a8_gemm_per_channel",
    op_func=mctlassEx_fused_moe_w4a8_gemm_per_channel,
    mutates_args=["c"],
    fake_impl=mctlassEx_fused_moe_w4a8_gemm_per_channel_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


#  get Kernel M
def mctlassEx_fused_moe_w4a8_get_kernel_m_per_channel(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    num_experts: int,
    batch_size: int,
    N: int,
    K: int,
    topk: int,
) -> int:
    assert mctlass_moe_gemm is not None, "mctlassOp is not imported correctly"
    return mctlass_moe_gemm.get_kernel_m(a, b, c, num_experts, batch_size, N, K, topk)


# end


def mctlassEx_fused_moe_w4a16_get_kernel_m(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    num_experts: int,
    batch_size: int,
    N: int,
    K: int,
    topk: int,
    group_size: int,
) -> int:
    assert mctlass_moe_gemm is not None, "mctlass op is not imported correctly"
    return mctlass_moe_gemm.get_kernel_m(
        a,
        b,
        c,
        num_experts,
        batch_size,
        N,
        K,
        topk,
        is_blockwise=True,
        group_size=group_size,
    )


def mctlassEx_fused_moe_w4a16_gemm(
    batch_size: int,
    N: int,
    K: int,
    num_experts: int,
    EM: int,
    topk: int,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    a_scales: torch.Tensor | None,
    b_scales: torch.Tensor,
    b_bias: torch.Tensor | None,
    topk_weights: torch.Tensor,
    token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    group_size: int,
    b_zp: torch.Tensor | None,
) -> torch.Tensor:
    assert mctlass_moe_gemm is not None, "mctlass op is not imported correctly"
    mctlass_moe_gemm(
        batch_size,
        N,
        K,
        num_experts,
        EM,
        topk,
        a,
        b,
        c,
        a_scales,
        b_scales,
        b_bias,
        topk_weights,
        token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight,
        is_blockwise=True,
        group_size=group_size,
        zp_b=b_zp,
    )
    return c


def mctlassEx_fused_moe_w4a16_gemm_fake(
    batch_size: int,
    N: int,
    K: int,
    num_experts: int,
    EM: int,
    topk: int,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    a_scales: torch.Tensor | None,
    b_scales: torch.Tensor,
    b_bias: torch.Tensor | None,
    topk_weights: torch.Tensor,
    token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
    group_size: int,
    b_zp: torch.Tensor | None,
) -> torch.Tensor:
    return c


direct_register_custom_op(
    op_name="mctlassEx_fused_moe_w4a16_gemm",
    op_func=mctlassEx_fused_moe_w4a16_gemm,
    mutates_args=["c"],
    fake_impl=mctlassEx_fused_moe_w4a16_gemm_fake,
    tags=(
        ()
        if is_torch_equal_or_newer("2.7.0")
        else (torch.Tag.needs_fixed_stride_order,)
    ),
)


def mctlassEx_fused_moe_bf16_get_kernel_m(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    num_experts: int,
    batch_size: int,
    N: int,
    K: int,
    topk: int,
) -> int:
    assert mctlass_moe_gemm is not None, "mctlass op is not imported correctly"
    kernel_m = mctlass_moe_gemm.get_kernel_m(
        A, B, C, num_experts, batch_size, N, K, topk
    )
    return kernel_m


def mctlassEx_fused_moe_bf16_gemm(
    batch_size: int,
    N: int,
    K: int,
    num_experts: int,
    EM: int,
    topk: int,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor,
    topk_weights: torch.Tensor,
    token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
) -> torch.Tensor:
    assert mctlass_moe_gemm is not None, "mctlass op is not imported correctly"
    mctlass_moe_gemm(
        batch_size,
        N,
        K,
        num_experts,
        EM,
        topk,
        A,
        B,
        C,
        scale_a,
        scale_b,
        bias,
        topk_weights,
        token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight,
    )
    return C


def mctlassEx_fused_moe_bf16_gemm_fake(
    batch_size: int,
    N: int,
    K: int,
    num_experts: int,
    EM: int,
    topk: int,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor,
    topk_weights: torch.Tensor,
    token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
) -> torch.Tensor:
    return C


direct_register_custom_op(
    op_name="mctlassEx_fused_moe_bf16_gemm",
    op_func=mctlassEx_fused_moe_bf16_gemm,
    mutates_args=["C"],
    fake_impl=mctlassEx_fused_moe_bf16_gemm_fake,
    tags=(
        ()
        if is_torch_equal_or_newer("2.7.0")
        else (torch.Tag.needs_fixed_stride_order,)
    ),
)


# w8a8 scaled mm
def mctlassEx_w8a8_scaled_mm_azp(
    out_dtype: torch.dtype,
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    azp_adj: torch.Tensor | None = None,
    azp: torch.Tensor | None = None,
) -> torch.Tensor:
    out = torch.empty((a.shape[0], b.shape[1]), dtype=out_dtype, device=a.device)
    if bias is not None and bias.dim() == 1:
        bias = bias.unsqueeze(0)

    assert mctlass_scaled_gemm is not None, "mctlass scale op is not imported correctly"
    _, K = a.shape
    M, N = out.shape
    mctlass_scaled_gemm(
        [M, N, K], a, b, out, scale_a, scale_b.T, bias, azp_adj=azp_adj, azp=azp
    )
    return out


def mctlassEx_w8a8_scaled_mm_azp_fake(
    out_dtype: torch.dtype,
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    azp_adj: torch.Tensor | None = None,
    azp: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty((a.shape[0], b.shape[1]), dtype=out_dtype, device=a.device)


direct_register_custom_op(
    op_name="mctlassEx_w8a8_scaled_mm_azp",
    op_func=mctlassEx_w8a8_scaled_mm_azp,
    mutates_args=[],
    fake_impl=mctlassEx_w8a8_scaled_mm_azp_fake,
    tags=(
        ()
        if is_torch_equal_or_newer("2.7.0")
        else (torch.Tag.needs_fixed_stride_order,)
    ),
)


# fp8 scaled mm
def mctlass_fp8_block_scaled_mm(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    M = A.shape[0]
    N = B.shape[0]
    K = B.shape[1]
    C = torch.zeros((M, N), dtype=out_dtype, device=A.device)
    scale_a_trans = As.T.contiguous()
    scale_b_trans = Bs.T.contiguous()
    assert mctlass_scaled_gemm is not None, "mctlass scale op is not imported correctly"
    mctlass_scaled_gemm(
        [M, N, K],
        A,
        B,
        C,
        scale_a_trans,
        scale_b_trans,
        None,
        is_blockwise=True,
        use_fp8=True,
        is_scale_a_1d=True,
        is_scale_b_1d=False,
        scale_a_layout="m-major",
        scale_b_layout="n-major",
    )
    return C


def mctlass_fp8_block_scaled_mm_fake(
    A: torch.Tensor,
    B: torch.Tensor,
    As: torch.Tensor,
    Bs: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    M = A.shape[0]
    N = B.shape[0]
    return torch.empty((M, N), dtype=out_dtype, device=A.device)


direct_register_custom_op(
    op_name="mctlass_fp8_block_scaled_mm",
    op_func=mctlass_fp8_block_scaled_mm,
    fake_impl=mctlass_fp8_block_scaled_mm_fake,
)


# w8a8 fused moe
def mctlassEx_fused_moe_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    bias: torch.Tensor | None,
    topk_weights: torch.Tensor,
    token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    EM: int,
    topk: int,
    mul_routed_weight: bool,
) -> torch.Tensor:
    assert mctlass_moe_gemm is not None, "mctlass op is not imported correctly"
    c1 = c.view(-1, c.size(-1)).contiguous()

    mctlass_moe_gemm(
        a.size(0),
        b.size(1),
        a.size(1),
        b.size(0),
        EM,
        topk,
        a,
        b,
        c1,
        a_scales,
        b_scales,
        bias,
        topk_weights,
        token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight,
    )
    return c1.reshape(c.shape)


def mctlassEx_fused_moe_gemm_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    bias: torch.Tensor | None,
    topk_weights: torch.Tensor,
    token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    EM: int,
    topk: int,
    mul_routed_weight: bool,
) -> torch.Tensor:
    return c


direct_register_custom_op(
    op_name="mctlassEx_fused_moe_gemm",
    op_func=mctlassEx_fused_moe_gemm,
    mutates_args=["c"],
    fake_impl=mctlassEx_fused_moe_gemm_fake,
    tags=(
        ()
        if is_torch_equal_or_newer("2.7.0")
        else (torch.Tag.needs_fixed_stride_order,)
    ),
)


# w4a8 fused moe
def mctlassEx_fused_moe_w4a8_get_kernel_m(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    num_experts: int,
    batch_size: int,
    N: int,
    K: int,
    num_valid_tokens: int,
    topk: int,
    group_size: int,
) -> int:
    assert mctlass_moe_gemm is not None, "mctlassMoeGEMM is not imported correctly"
    return mctlass_moe_gemm.get_kernel_m(
        a,
        b,
        c,
        num_experts,
        batch_size,
        N,
        K,
        topk,
        num_valid_tokens=num_valid_tokens,
        is_blockwise=True,
        group_size=group_size,
    )


def mctlassEx_fused_moe_w4a8_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    topk_weights: torch.Tensor,
    token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    num_experts: int,
    batch_size: int,
    N: int,
    K: int,
    num_valid_tokens: int,
    EM: int,
    topk: int,
    mul_routed_weight: bool,
    group_size: int,
) -> torch.Tensor:
    assert mctlass_moe_gemm is not None, "mctlassMoeGEMM is not imported correctly"
    mctlass_moe_gemm(
        batch_size,
        N,
        K,
        num_experts,
        EM,
        topk,
        a,
        b,
        c,
        a_scales,
        b_scales,
        None,
        topk_weights,
        token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight,
        is_blockwise=True,
        group_size=group_size,
    )
    return c


def mctlassEx_fused_moe_w4a8_gemm_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    topk_weights: torch.Tensor,
    token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    num_experts: int,
    batch_size: int,
    N: int,
    K: int,
    num_valid_tokens: int,
    EM: int,
    topk: int,
    mul_routed_weight: bool,
    group_size: int,
) -> torch.Tensor:
    return c


direct_register_custom_op(
    op_name="mctlassEx_fused_moe_w4a8_gemm",
    op_func=mctlassEx_fused_moe_w4a8_gemm,
    mutates_args=["c"],
    fake_impl=mctlassEx_fused_moe_w4a8_gemm_fake,
    tags=(
        ()
        if is_torch_equal_or_newer("2.7.0")
        else (torch.Tag.needs_fixed_stride_order,)
    ),
)


def cutlass_moe_w4a8_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    topk_weights: torch.Tensor,
    token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    num_experts: int,
    batch_size: int,
    N: int,
    K: int,
    num_valid_tokens: int,
    EM: int,
    topk: int,
    mul_routed_weight: bool,
    group_size: int,
) -> torch.Tensor:
    return torch.ops.vllm.mctlassEx_fused_moe_w4a8_gemm(
        a,
        b,
        c,
        a_scales,
        b_scales,
        topk_weights,
        token_ids,
        expert_ids,
        num_tokens_post_padded,
        num_experts,
        batch_size,
        N,
        K,
        num_valid_tokens,
        EM,
        topk,
        mul_routed_weight,
        group_size,
    )


# w8a8 fp8 fused moe
def mctlassEx_fused_moe_w8a8_fp8_get_kernel_m(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    topk: int,
    block_shape: list[int] | None = None,
) -> int:
    assert mctlass_moe_gemm is not None, "mctlassMoeGEMM is not imported correctly"
    c1 = c.view(-1, c.size(-1))
    assert c1.is_contiguous(), "fused moe output buffer is not contiguous"
    kernel_m_kwargs: dict[str, Any] = {"use_fp8": True}
    if block_shape is not None:
        kernel_m_kwargs.update(
            is_blockwise=True,
            group_size=block_shape[0],
        )
    return mctlass_moe_gemm.get_kernel_m(
        a,
        b,
        c1,
        b.shape[0],  # num_experts
        a.shape[0],  # batch_size
        b.shape[1],  # N
        a.shape[1],  # k
        topk,
        **kernel_m_kwargs,
    )


def mctlassEx_fused_moe_w8a8_fp8_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    topk_weights: torch.Tensor,
    token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    EM: int,
    topk: int,
    mul_routed_weight: bool,
    block_shape: list[int] | None = None,
    filter_expert: bool = True,
) -> None:
    assert mctlass_moe_gemm is not None, "mctlassMoeGEMM is not imported correctly"
    c1 = c.view(-1, c.size(-1))
    assert c1.is_contiguous(), "fused moe output buffer is not contiguous"
    fp8_kwargs: dict[str, Any] = {
        "filter_expert": filter_expert,
        "use_fp8": True,
    }
    if block_shape is not None:
        a_scales = a_scales.T.contiguous()
        b_scales = b_scales.transpose(1, 2).contiguous()
        fp8_kwargs.update(
            is_blockwise=True,
            group_size=block_shape[0],
            is_scale_a_1d=True,
            is_scale_b_1d=False,
            scale_a_layout="m-major",
            scale_b_layout="n-major",
        )
    mctlass_moe_gemm(
        a.shape[0],  # m
        b.shape[1],  # n
        a.shape[1],  # k
        b.shape[0],  # num_expert
        EM,
        topk,
        a,
        b,
        c1,
        a_scales,
        b_scales,
        None,  # bias
        topk_weights,
        token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight,
        **fp8_kwargs,
    )


def mctlassEx_fused_moe_w8a8_fp8_gemm_fake(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    topk_weights: torch.Tensor,
    token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    EM: int,
    topk: int,
    mul_routed_weight: bool,
    block_shape: list[int] | None = None,
    filter_expert: bool = True,
) -> None:
    return


direct_register_custom_op(
    op_name="mctlassEx_fused_moe_w8a8_fp8",
    op_func=mctlassEx_fused_moe_w8a8_fp8_gemm,
    mutates_args=["c"],
    fake_impl=mctlassEx_fused_moe_w8a8_fp8_gemm_fake,
    tags=(
        ()
        if is_torch_equal_or_newer("2.7.0")
        else (torch.Tag.needs_fixed_stride_order,)
    ),
)


def cutlass_moe_w8a8_fp8(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    topk_weights: torch.Tensor,
    token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    EM: int,
    topk: int,
    mul_routed_weight: bool,
    block_shape: list[int] | None = None,
) -> torch.Tensor:
    torch.ops.vllm.mctlassEx_fused_moe_w8a8_fp8(
        a,
        b,
        c,
        a_scales,
        b_scales,
        topk_weights,
        token_ids,
        expert_ids,
        num_tokens_post_padded,
        EM,
        topk,
        mul_routed_weight,
        block_shape,
    )

    return c


# -------------------------------------------------
# Note:
#
# This is different from `cutlass_scaled_mm` in `_cutlass_ops.py`.
# It invokes mctlassEx python API directly.
# -------------------------------------------------
def cutlass_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
    assert bias is None or bias.numel() == b.shape[1] and bias.dtype == out_dtype

    # Massage the input to be 2D
    target_shape = (*a.shape[:-1], b.shape[1])
    a = a.view(-1, a.shape[-1])

    # out = torch.empty((a.shape[0], b.shape[1]), dtype=out_dtype, device=a.device)

    out = torch.ops.vllm.mctlassEx_w8a8_scaled_mm_azp(
        out_dtype, a, b, scale_a, scale_b, bias
    )

    return out.view(*target_shape)


# -------------------------------------------------
# Note:
#
# This is different from `cutlass_scaled_mm_azp` in `_cutlass_ops.py`.
# It invokes mctlassEx python API directly.
# -------------------------------------------------
def cutlass_scaled_mm_azp(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    azp_adj: torch.Tensor,
    azp: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    assert b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
    assert bias is None or bias.numel() == b.shape[1] and bias.dtype == out_dtype

    # Massage the input to be 2D
    target_shape = (*a.shape[:-1], b.shape[1])
    a = a.view(-1, a.shape[-1])
    assert azp is None or azp.numel() == a.shape[0]

    # out = torch.empty((a.shape[0], b.shape[1]), dtype=out_dtype, device=a.device)
    out = torch.ops.vllm.mctlassEx_w8a8_scaled_mm_azp(
        out_dtype, a, b, scale_a, scale_b, bias, azp_adj, azp
    )

    return out.view(*target_shape)


# -------------------------------------------------
# Note:
#
# This is only supported in `_python_api_ops.py`.
# It invokes mctlassEx python API directly.
# -------------------------------------------------
def cutlass_fp8_block_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return torch.ops.vllm.mctlass_fp8_block_scaled_mm(a, b, scale_a, scale_b, out_dtype)


# -------------------------------------------------
# Note:
#
# This is different from `cutlass_moe_mm_w8a8_get_kernel_m` in `_cutlass_ops.py`.
# It invokes mctlassEx python API directly.
# -------------------------------------------------
def cutlass_moe_mm_w8a8_get_kernel_m(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, topk: int
) -> int:
    assert mctlass_moe_gemm is not None, "mctlass op is not imported correctly"
    qa = a.to(torch.int8)
    qb = b.to(torch.int8)
    c1 = c.view(-1, c.size(-1)).contiguous()  # noqa: F841
    batch_size = qa.size(0)
    K = qa.size(1)
    num_experts = qb.size(0)
    N = qb.size(1)
    return mctlass_moe_gemm.get_kernel_m(
        qa, qb, c1, num_experts, batch_size, N, K, topk
    )


# -------------------------------------------------
# Note:
#
# This is different from `cutlass_moe_mm_w8a8` in `_cutlass_ops.py`.
# It invokes mctlassEx python API directly.
# -------------------------------------------------
def cutlass_moe_mm_w8a8(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    bias: torch.Tensor | None,
    moe_weight: torch.Tensor,
    token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    EM: int,
    topk: int,
    mul_routed_weight: bool,
) -> torch.Tensor:
    assert bias is None, "mctlass api not support w8a8 with bias currently."
    torch.ops.vllm.mctlassEx_fused_moe_gemm(
        a,
        b,
        c,
        a_scales,
        b_scales,
        bias,
        moe_weight,
        token_ids,
        expert_ids,
        num_tokens_post_padded,
        EM,
        topk,
        mul_routed_weight,
    )


# support W4A8 Per-Channel  start
# Kernel M
def cutlass_moe_mm_w4a8_get_kernel_m_per_channel(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    K: int,
    num_valid_tokens: int,
    topk: int,
) -> int:
    batch_size = a.size(0)
    num_experts, N, _ = b.size()

    # a to int8
    a = a.to(torch.int8)
    b = b.view(dtype=torch.quint4x2)

    return mctlassEx_fused_moe_w4a8_get_kernel_m_per_channel(
        a=a,
        b=b,
        c=c,
        num_experts=num_experts,
        batch_size=batch_size,
        N=N,
        K=K,
        topk=topk,
    )


# support W4A8 Per-Channel
# GEMM
def cutlass_moe_mm_w4a8_per_channel(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    b_bias: torch.Tensor,
    topk_weights: torch.Tensor,
    token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    EM: int,
    topk: int,
    mul_routed_weight: bool,
) -> torch.Tensor:
    batch_size = a.size(0)
    num_experts = b.size(0)
    N = b.size(1)
    K = b.size(2) * 8

    return torch.ops.vllm.mctlassEx_fused_moe_w4a8_gemm_per_channel(
        batch_size=batch_size,
        N=N,
        K=K,
        num_experts=num_experts,
        EM=EM,
        topk=topk,
        a=a,
        b=b.view(dtype=torch.quint4x2),
        c=c,
        a_scales=a_scales,
        b_scales=b_scales,
        b_bias=b_bias,
        topk_weights=topk_weights,
        token_ids=token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=mul_routed_weight,
    )


# end


def cutlass_moe_mm_bf16(
    batch_size: int,
    N: int,
    K: int,
    num_experts: int,
    EM: int,
    topk: int,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor,
    topk_weights: torch.Tensor,
    token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    mul_routed_weight: bool,
) -> torch.Tensor:
    return mctlassEx_fused_moe_bf16_gemm(
        batch_size,
        N,
        K,
        num_experts,
        EM,
        topk,
        A,
        B,
        C,
        scale_a,
        scale_b,
        bias,
        topk_weights,
        token_ids,
        expert_ids,
        num_tokens_post_padded,
        mul_routed_weight,
    )


def cutlass_moe_mm_w4a16_get_kernel_m(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    K: int,
    num_valid_tokens: int,
    topk: int,
    group_size: int,
) -> int:
    batch_size = a.size(0)
    num_experts, N, _ = b.size()

    return mctlassEx_fused_moe_w4a16_get_kernel_m(
        a=a,
        b=b.view(dtype=torch.quint4x2),
        c=c,
        num_experts=num_experts,
        batch_size=batch_size,
        N=N,
        K=K,
        topk=topk,
        group_size=group_size,
    )


def cutlass_moe_mm_w4a16(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    b_scales: torch.Tensor,
    b_zp: torch.Tensor,
    topk_weights: torch.Tensor,
    token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    topk: int,
    mul_routed_weight: bool,
    group_size: int,
) -> torch.Tensor:
    batch_size = a.size(0)
    K = a.size(1)
    num_experts, N, _ = b.size()
    EM = token_ids.size(0)

    return torch.ops.vllm.mctlassEx_fused_moe_w4a16_gemm(
        batch_size=batch_size,
        N=N,
        K=K,
        num_experts=num_experts,
        EM=EM,
        topk=topk,
        a=a,
        b=b.view(dtype=torch.quint4x2),
        c=c,
        a_scales=None,
        b_scales=b_scales,
        b_bias=None,
        topk_weights=topk_weights,
        token_ids=token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        mul_routed_weight=mul_routed_weight,
        group_size=group_size,
        b_zp=b_zp,
    )
