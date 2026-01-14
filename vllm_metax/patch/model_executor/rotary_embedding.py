# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

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


# suport Qwen3-omni
from typing import Optional, Union
from transformers import PretrainedConfig
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding


class MacaMRotaryEmbedding(MRotaryEmbedding):
    """Rotary Embedding with Multimodal Sections."""

    @classmethod
    def get_input_positions_tensor(
        cls,
        input_tokens: list[int],
        hf_config: PretrainedConfig,
        image_grid_thw: Union[list[list[int]], torch.Tensor],
        video_grid_thw: Union[list[list[int]], torch.Tensor],
        second_per_grid_ts: list[float],
        context_len: int = 0,
        seq_len: Optional[int] = None,
        audio_feature_lengths: Optional[torch.Tensor] = None,
        use_audio_in_video: bool = False,
    ) -> tuple[torch.Tensor, int]:
        from vllm.transformers_utils.config import thinker_uses_mrope

        if thinker_uses_mrope(hf_config) and hf_config.model_type == "qwen2_5_omni":
            return cls._omni_get_input_positions_tensor(
                input_tokens=input_tokens,
                hf_config=hf_config,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                context_len=context_len,
                seq_len=seq_len,
                audio_feature_lengths=audio_feature_lengths,
                use_audio_in_video=use_audio_in_video,
            )
        elif hf_config.model_type in ["glm4v", "glm4v_moe"]:
            return cls._glm4v_get_input_positions_tensor(
                input_tokens=input_tokens,
                hf_config=hf_config,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                context_len=context_len,
                seq_len=seq_len,
            )
        elif hf_config.model_type in ["qwen3_vl", "qwen3_vl_moe"]:
            return cls._qwen3vl_get_input_positions_tensor(
                input_tokens=input_tokens,
                hf_config=hf_config,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                context_len=context_len,
                seq_len=seq_len,
            )
        elif hf_config.model_type in ["ernie4_5_moe_vl", "ernie4_5_vl"]:
            return cls._ernie_get_input_positions_tensor(
                input_tokens=input_tokens,
                hf_config=hf_config,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                context_len=context_len,
                seq_len=seq_len,
            )
        elif "KeyeVL1_5" in hf_config.model_type:
            return cls._keye_get_input_positions_tensor(
                input_tokens=input_tokens,
                hf_config=hf_config,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                context_len=context_len,
                seq_len=seq_len,
            )
        else:
            return cls._vl_get_input_positions_tensor(
                input_tokens=input_tokens,
                hf_config=hf_config,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                context_len=context_len,
                seq_len=seq_len,
            )


import vllm.model_executor.layers.rotary_embedding.mrope

vllm.model_executor.layers.rotary_embedding.mrope.MRotaryEmbedding.get_input_positions_tensor = MacaMRotaryEmbedding.get_input_positions_tensor
