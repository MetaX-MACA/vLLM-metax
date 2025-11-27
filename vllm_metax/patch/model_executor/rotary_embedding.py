# SPDX-License-Identifier: Apache-2.0

# -----------------------------------------------
# Note: Use maca's flash_attn instead of vllm's
# -----------------------------------------------

import vllm
from vllm.logger import init_logger

logger = init_logger(__name__)

import torch


def apply_rotary_emb_dispatch(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, is_neox_style: bool
) -> torch.Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    # ┌------------------------  Metax Modification --------------------------------┐
    from flash_attn.layers.rotary import apply_rotary_emb

    return apply_rotary_emb(x.unsqueeze(0), cos, sin, not is_neox_style).squeeze(0)
    # └------------------------- Metax Modification --------------------------------┘


import vllm.model_executor.layers.rotary_embedding.common

vllm.model_executor.layers.rotary_embedding.common.apply_rotary_emb_dispatch = (
    apply_rotary_emb_dispatch
)
