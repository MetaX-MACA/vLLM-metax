# SPDX-License-Identifier: Apache-2.0
# ---------------------------------------------------
# Note: enable with MACA_VLLM_ENABLE_MCTLASS_PYTHON_API=1
# ---------------------------------------------------

import torch
import contextlib
from vllm.utils.torch_utils import direct_register_custom_op, is_torch_equal_or_newer
from vllm.logger import logger

mctlass_op = None
mctlass_scaled_gemm = None
with contextlib.suppress(ImportError):
    if mctlass_op is None:
        import mctlassEx

        mctlass_op = mctlassEx.mctlassExHandleWrapper()

        try:
            from mctlassEx import ScaledGEMM

            mctlass_scaled_gemm = ScaledGEMM()
        except ImportError:
            logger.warning(
                "Failed to import ScaledGEMM from mctlass. "
                "scaled_mm_azp not support for now"
            )


# w8a8 scaled mm
def mctlassEx_w8a8_scaled_mm_azp(
    out: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    azp_adj: torch.Tensor | None = None,
    azp: torch.Tensor | None = None,
) -> torch.Tensor:
    if bias is not None and bias.dim() == 1:
        bias = bias.unsqueeze(0)

    if azp is not None or azp_adj is not None or bias is not None:
        assert mctlass_scaled_gemm is not None, (
            "scaled_mm with azp and bias is not supported for current mctlass version"
        )
        _, K = a.shape
        M, N = out.shape
        mctlass_scaled_gemm(
            [M, N, K], a, b, out, scale_a, scale_b.T, bias, azp_adj=azp_adj, azp=azp
        )
    else:
        assert mctlass_op is not None, "mctlassOp is not imported correctly"
        stream = torch.cuda.current_stream().cuda_stream
        mctlass_op.mctlass_w8a8_scaled_mm_azp(
            a, b, out, scale_a, scale_b.T, bias, azp_adj, azp, stream
        )
    return out


def mctlassEx_w8a8_scaled_mm_azp_fake(
    out: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    azp_adj: torch.Tensor | None = None,
    azp: torch.Tensor | None = None,
) -> torch.Tensor:
    return out


direct_register_custom_op(
    op_name="mctlassEx_w8a8_scaled_mm_azp",
    op_func=mctlassEx_w8a8_scaled_mm_azp,
    mutates_args=["out"],
    fake_impl=mctlassEx_w8a8_scaled_mm_azp_fake,
    tags=(
        ()
        if is_torch_equal_or_newer("2.7.0")
        else (torch.Tag.needs_fixed_stride_order,)
    ),
)


# w8a8 fused moe
def mctlassEx_fused_moe_gemm(
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
) -> torch.Tensor:
    # TODO: need mctlass to fix it
    stream = torch.cuda.current_stream().cuda_stream
    c1 = c.view(-1, c.size(-1)).contiguous()
    assert mctlass_op is not None, "mctlassOp is not imported correctly"
    mctlass_op.mctlass_fuse_moe_gemm(
        a,
        b,
        c1,
        a_scales,
        b_scales,
        topk_weights,
        token_ids,
        expert_ids,
        num_tokens_post_padded,
        EM,
        topk,
        mul_routed_weight,
        stream,
    )
    return c1.reshape(c.shape)


def mctlassEx_fused_moe_gemm_fake(
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
    assert mctlass_op is not None, "mctlassOp is not imported correctly"
    mctlass_op.mctlass_fuse_moe_gemm_basic(
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

    out = torch.empty((a.shape[0], b.shape[1]), dtype=out_dtype, device=a.device)

    torch.ops.vllm.mctlassEx_w8a8_scaled_mm_azp(out, a, b, scale_a, scale_b, bias)

    return out.view(*target_shape)


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

    out = torch.empty((a.shape[0], b.shape[1]), dtype=out_dtype, device=a.device)
    torch.ops.vllm.mctlassEx_w8a8_scaled_mm_azp(
        out, a, b, scale_a, scale_b, bias, azp_adj, azp
    )

    return out.view(*target_shape)


def cutlass_moe_mm_w8a8_get_kernel_m(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, topk: int
) -> int:
    assert mctlass_op is not None, "mctlassOp is not imported correctly"
    qa = a.to(torch.int8)
    qb = b.to(torch.int8)
    c1 = c.view(-1, c.size(-1)).contiguous()
    return mctlass_op.mctlass_fuse_moe_get_kernel_m(qa, qb, c1, topk)


def cutlass_moe_mm_w8a8(
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    a_scales: torch.Tensor,
    b_scales: torch.Tensor,
    moe_weight: torch.Tensor,
    token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    EM: int,
    topk: int,
    mul_routed_weight: bool,
) -> torch.Tensor:
    torch.ops.vllm.mctlassEx_fused_moe_gemm(
        a,
        b,
        c,
        a_scales,
        b_scales,
        moe_weight,
        token_ids,
        expert_ids,
        num_tokens_post_padded,
        EM,
        topk,
        mul_routed_weight,
    )
