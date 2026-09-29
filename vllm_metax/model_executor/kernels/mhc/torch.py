# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch


def expand_to_mhc_ref(hidden: torch.Tensor, mhc_mult: int) -> torch.Tensor:
    return (
        hidden.unsqueeze(-2)
        .expand(*hidden.shape[:-1], mhc_mult, hidden.shape[-1])
        .contiguous()
    )


def sinkhorn_normalize_ref(
    x: torch.Tensor, repeat: int = 10, eps: float = 1e-6
) -> torch.Tensor:
    x = x.softmax(-1) + eps
    x = x / (x.sum(-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        x = x / (x.sum(-1, keepdim=True) + eps)
        x = x / (x.sum(-2, keepdim=True) + eps)
    return x


def mhc_head_compute_mix_ref(
    input_mix: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    mhc_pre_eps: float,
) -> torch.Tensor:
    mhc_head_layer_mix = input_mix * mhc_scale + mhc_base
    return torch.sigmoid(mhc_head_layer_mix) + mhc_pre_eps


def mhc_pre_split_mixes_ref(
    input_mixes: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    mhc_mult: int,
    mhc_post_mult_value: float,
    mhc_pre_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    a, b = input_mixes.shape[:2]
    mhc_scale = torch.cat(
        [
            mhc_scale[0].expand(mhc_mult),
            mhc_scale[1].expand(mhc_mult),
            mhc_scale[2].expand(mhc_mult * mhc_mult),
        ],
    )
    # print(f">>>>>>>>>>>> mhc_pre_split_mixes_ref, input_mixes: {input_mixes.shape}, mhc_scale: {mhc_scale.shape}, mhc_base: {mhc_base.shape}")
    input_mixes = input_mixes * mhc_scale + mhc_base

    pre_layer_mix = input_mixes[:, :, :mhc_mult].sigmoid().unsqueeze(-1) + mhc_pre_eps
    post_layer_mix = (
        input_mixes[:, :, mhc_mult : 2 * mhc_mult].sigmoid() * mhc_post_mult_value
    ).unsqueeze(-1)
    comb_res_mix = input_mixes[:, :, 2 * mhc_mult :].view(a, b, mhc_mult, mhc_mult)

    return pre_layer_mix, post_layer_mix, comb_res_mix


def mhc_pre_apply_mix_ref(x: torch.Tensor, mix: torch.Tensor) -> torch.Tensor:
    return (x * mix).sum(-2).bfloat16()


def mhc_pre_norm_fn_ref(
    residual: torch.Tensor,
    mhc_fn: torch.Tensor,
    mhc_norm_weight: torch.Tensor | None,
    mhc_norm_eps: float,
) -> torch.Tensor:
    if mhc_norm_weight is not None:
        mhc_fn = mhc_fn * mhc_norm_weight
    residual = residual.unsqueeze(0)
    residual = residual.flatten(2, 3).float()
    assert mhc_fn.dtype == residual.dtype == torch.float
    mhc_mult = mhc_fn.shape[0]
    rms_group_size = mhc_fn.shape[-1]
    mixes = torch.einsum(
        "mbk,nbk->mbn",
        residual.view(-1, 1, rms_group_size),
        mhc_fn.view(mhc_mult, 1, rms_group_size),
    )
    sqrsum = residual.view(-1, 1, rms_group_size).square().sum(-1)
    mixes = (
        mixes * (sqrsum.unsqueeze(-1) / rms_group_size + mhc_norm_eps).rsqrt()
    ).sum(-2)
    return mixes.view(*residual.shape[:2], -1)


def big_fuse_reference(
    residual: torch.Tensor,
    fn: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mhc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    outer_shape = residual.shape[:-2]

    mixes = mhc_pre_norm_fn_ref(
        residual,
        fn,
        None,
        rms_eps,
        # fuse_grad_acc=False,
        # n_splits=n_splits,
    )

    pre_mix, post_mix, comb_mix = mhc_pre_split_mixes_ref(
        mixes,
        mhc_scale,
        mhc_base,
        mhc_mult,
        mhc_post_mult_value,
        mhc_pre_eps,
    )

    comb_mix = sinkhorn_normalize_ref(
        comb_mix, repeat=sinkhorn_repeat, eps=mhc_sinkhorn_eps
    )

    layer_input = mhc_pre_apply_mix_ref(residual, pre_mix)

    post_mix = post_mix.view(*outer_shape, mhc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, mhc_mult, mhc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)

    return post_mix, comb_mix, layer_input


def mhc_pre_delayed_torch(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    pre_mix: torch.Tensor | None = None,
    x: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference for mHC pre using coefficients from the previous sublayer."""
    hc_mult = residual.shape[1]
    x = (residual.flatten(1) if x is None else x).float()
    mixes = (x @ fn.t()) * torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_eps)
    pre = (
        torch.sigmoid(mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + hc_pre_eps
    )
    post = (
        torch.sigmoid(
            mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1]
            + hc_base[hc_mult : 2 * hc_mult]
        )
        * hc_post_mult_value
    )
    comb = mixes[:, 2 * hc_mult :].view(-1, hc_mult, hc_mult) * hc_scale[2]
    comb = comb + hc_base[2 * hc_mult :].view(1, hc_mult, hc_mult)
    comb = torch.softmax(comb, dim=-1) + hc_sinkhorn_eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    layer_input = (
        residual[:, 0]
        if pre_mix is None
        else (pre_mix.unsqueeze(-1) * residual.float()).sum(dim=1).to(residual.dtype)
    )
    return post.unsqueeze(-1), comb, layer_input, pre


def mhc_pre_torch(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward pass for mHC pre block.

    Args:
        residual: shape (..., hc_mult, hidden_size), dtype torch.bfloat16
        fn: shape (hc_mult3, hc_mult * hidden_size), dtype torch.float32
        hc_scale: shape (3,), dtype torch.float32
        hc_base: shape (hc_mult3,), dtype torch.float32
        rms_eps: RMS normalization epsilon
        hc_pre_eps: pre-mix epsilon
        hc_sinkhorn_eps: sinkhorn epsilon
        hc_post_mult_value: post-mix multiplier value
        sinkhorn_repeat: number of sinkhorn iterations
        n_splits: split-k factor;

    Returns:
        post_mix: shape (..., hc_mult), dtype torch.float32
        comb_mix: shape (..., hc_mult, hc_mult), dtype torch.float32
        layer_input: shape (..., hidden_size), dtype torch.bfloat16
    """

    # Validate shapes
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2

    hc_hidden_size = hc_mult * hidden_size
    assert fn.shape[0] == hc_mult3
    assert fn.shape[1] == hc_hidden_size
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    fn_flat = fn

    x = residual_flat.view(num_tokens, hc_mult * hidden_size).to(torch.float32)
    mixes = torch.matmul(x, fn_flat.t())
    sqrsum = x.square().sum(dim=-1, keepdim=True)
    mixes = mixes * torch.rsqrt(sqrsum / (hc_mult * hidden_size) + rms_eps)

    pre_logits = mixes[:, :hc_mult] * hc_scale[0] + hc_base[:hc_mult]
    pre_mix = torch.sigmoid(pre_logits) + hc_pre_eps

    post_logits = (
        mixes[:, hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    )
    post_mix = torch.sigmoid(post_logits) * hc_post_mult_value

    comb_logits = mixes[:, 2 * hc_mult :].view(num_tokens, hc_mult, hc_mult) * hc_scale[
        2
    ] + hc_base[2 * hc_mult :].view(1, hc_mult, hc_mult)
    comb_mix = torch.softmax(comb_logits, dim=-1) + hc_sinkhorn_eps
    comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_repeat - 1):
        comb_mix = comb_mix / (comb_mix.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        comb_mix = comb_mix / (comb_mix.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    layer_input = torch.sum(
        pre_mix.unsqueeze(-1) * residual_flat.to(torch.float32), dim=1
    ).to(torch.bfloat16)
    return (
        post_mix.view(*outer_shape, hc_mult, 1),
        comb_mix.view(*outer_shape, hc_mult, hc_mult),
        layer_input.view(*outer_shape, hidden_size),
    )


def mhc_post_torch(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    mixed_residual = torch.einsum(
        "...ij,...ih->...jh",
        comb_res_mix.to(torch.float32),
        residual.to(torch.float32),
    )
    post_term = post_layer_mix.to(torch.float32) * x.unsqueeze(-2).to(torch.float32)
    return (mixed_residual + post_term).to(residual.dtype)
