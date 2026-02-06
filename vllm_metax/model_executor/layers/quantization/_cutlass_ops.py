# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

import torch
from vllm.platforms import current_platform


# scale mm
def cutlass_scaled_mm(
    a: torch.Tensor,
    b: torch.Tensor,
    scale_a: torch.Tensor,
    scale_b: torch.Tensor,
    out_dtype: torch.dtype,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    `cutlass_scaled_mm` implements a fused version of
        `output = torch.mm((scale_a * a), (scale_b * b)).to(out_dtype)`
    where scale_a * a and scale_b * b are implemented using numpy-style
    broadcasting.

    In order to support blockwise scaling like found in DeepSeek V3 we also
    support extended "group" broadcast rules. We extend the numpy-style
    broadcasting rules with the following rule:
        "if the extent of a dimension in the source shape is between 1 and
        corresponding extent in the target shape we repeat each element along
        that dimension  src_shape[dim] // target_shape[dim] times consecutively"
    example if we have:
          a = [[1, 2], and target_shape = (2, 4)
               [3, 4]]
    then we would expand a to:
          a = [[1, 1, 2, 2],
               [3, 3, 4, 4]]
    currently we only support the case:
        scale_a.shape * [1, 128] == a.shape
        scale_b.shape * [128, 128] == b.shape
    """
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
    assert bias is None or bias.numel() == b.shape[1] and bias.dtype == out_dtype

    # Massage the input to be 2D
    target_shape = (*a.shape[:-1], b.shape[1])
    a = a.view(-1, a.shape[-1])

    cutlass_compatible_b = b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0
    if current_platform.is_rocm() or not cutlass_compatible_b:
        from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa
            triton_scaled_mm,
        )

        out = triton_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias)
    else:
        out = torch.empty((a.shape[0], b.shape[1]), dtype=out_dtype, device=a.device)
        torch.ops._C.cutlass_scaled_mm(out, a, b, scale_a, scale_b, bias)

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
    """
    :param azp_adj: In the per-tensor case, this should include the azp.
    Always per-channel.
    :param azp: Only set in the per-token case. Per-token if set.
    """
    assert b.shape[0] % 16 == 0 and b.shape[1] % 16 == 0
    assert out_dtype is torch.bfloat16 or out_dtype is torch.float16
    assert bias is None or bias.numel() == b.shape[1] and bias.dtype == out_dtype

    # Massage the input to be 2D
    target_shape = (*a.shape[:-1], b.shape[1])
    a = a.view(-1, a.shape[-1])
    assert azp is None or azp.numel() == a.shape[0]

    out = torch.empty((a.shape[0], b.shape[1]), dtype=out_dtype, device=a.device)
    torch.ops._C.cutlass_scaled_mm_azp(out, a, b, scale_a, scale_b, azp_adj, azp, bias)
    return out.view(*target_shape)


# w8a8 fused moe
def cutlass_moe_mm_w8a8_get_kernel_m(
    a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, topk: int
) -> int:
    num_valid_tokens = topk * c.shape[0]
    N, group, _ = b.size()
    K = a.shape[1]
    return torch.ops._C.cutlass_moe_mm_gemm_kernel_m_w8a8(num_valid_tokens, N, K, group)


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
    N = b.shape[1]
    K = a.shape[1]
    num_valid_tokens = a.size(0) * topk
    torch.ops._C.cutlass_moe_mm_w8a8(
        a,
        b,
        c,
        a_scales,
        b_scales,
        moe_weight,
        token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        EM,
        num_valid_tokens,
        topk,
        mul_routed_weight,
    )


# bf16 fused moe
def cutlass_moe_bf16_mm(
    out: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    moe_weight: torch.Tensor,
    token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    num_valid_tokens: int,
    topk: int,
    mul_routed_weight: bool,
) -> torch.Tensor:
    return torch.ops._C.cutlass_moe_bf16_mm(
        out,
        a,
        b,
        moe_weight,
        token_ids,
        expert_ids,
        num_tokens_post_padded,
        num_valid_tokens,
        topk,
        mul_routed_weight,
    )
