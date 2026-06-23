# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch
import torch.nn as nn

from vllm_metax.utils.deep_gemm import bf16_einsum
from . import inv_rope


def deep_gemm_bf16_o_proj(
    o: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    wo_a: nn.Module,
    wo_b: nn.Module,
    *,
    n_groups: int,
    heads_per_group: int,
    nope_dim: int,
    rope_dim: int,
    o_lora_rank: int,
) -> torch.Tensor:
    """
    O projection: inverse RoPE + einsum + wo_b.

    """
    o_bf16 = inv_rope(
        o,
        positions,
        cos_sin_cache,
        n_groups=n_groups,
        heads_per_group=heads_per_group,
        nope_dim=nope_dim,
        rope_dim=rope_dim,
    )
    wo_a_bf16 = wo_a.weight.view(n_groups, o_lora_rank, -1)
    z = torch.empty(
        (o.shape[0], n_groups, o_lora_rank),
        device=o.device,
        dtype=torch.bfloat16,
    )
    bf16_einsum(
        "bhr,hdr->bhd",
        o_bf16,
        wo_a_bf16,
        z,
    )
    return wo_b(z.flatten(1))
