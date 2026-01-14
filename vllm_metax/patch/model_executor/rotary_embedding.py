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

    # @classmethod
    # def _keye_get_input_positions_tensor(
    #     cls,
    #     input_tokens: list[int],
    #     hf_config: PretrainedConfig,
    #     image_grid_thw: Union[list[list[int]], torch.Tensor],
    #     video_grid_thw: Union[list[list[int]], torch.Tensor],
    #     context_len: int = 0,
    #     seq_len: Optional[int] = None,
    # ) -> tuple[torch.Tensor, int]:
    #     if isinstance(video_grid_thw, list) and len(video_grid_thw) > 0:
    #         video_grid_thw = video_grid_thw[0]
    #     """Get mrope input positions and delta value (Keye series)."""

    #     def split_thw(grid_thw: Union[torch.Tensor, list[int]]) -> list[list[int]]:
    #         """
    #         Split grid_thw along the t dimension.

    #         Args:
    #             grid_thw: shape [N, 3] tensor or nested list of [t, h, w].

    #         Returns:
    #             List of [1, h, w] rows, repeated t times for each original row.
    #         """

    #         if isinstance(grid_thw, list):
    #             grid_thw = torch.tensor(grid_thw, dtype=torch.long)

    #         if grid_thw.numel() == 0:
    #             return []

    #         t, hw = grid_thw[:, 0], grid_thw[:, 1:]
    #         ones = torch.ones_like(hw[:, :1])  # [N,1]
    #         out = torch.cat([ones, hw], dim=1).repeat_interleave(t, dim=0)
    #         return out.tolist()

    #     video_grid_thw = split_thw(video_grid_thw)

    #     image_token_id = hf_config.image_token_id
    #     video_token_id = hf_config.video_token_id
    #     spatial_merge_size = hf_config.vision_config.spatial_merge_size

    #     image_nums = len(image_grid_thw)
    #     frame_nums = len(video_grid_thw)
    #     llm_pos_ids_list: list = []

    #     st = 0
    #     remain_images, remain_frames = image_nums, frame_nums

    #     image_index, video_index = 0, 0
    #     for _ in range(image_nums + frame_nums):
    #         if remain_images > 0:
    #             try:
    #                 ed_image = input_tokens.index(image_token_id, st)
    #             except ValueError:
    #                 ed_image = len(input_tokens) + 1
    #         else:
    #             ed_image = len(input_tokens) + 1
    #         if remain_frames > 0:
    #             try:
    #                 ed_video = input_tokens.index(video_token_id, st)
    #             except ValueError:
    #                 ed_video = len(input_tokens) + 1
    #         else:
    #             ed_video = len(input_tokens) + 1

    #         if ed_image < ed_video:
    #             t, h, w = (
    #                 image_grid_thw[image_index][0],
    #                 image_grid_thw[image_index][1],
    #                 image_grid_thw[image_index][2],
    #             )
    #             image_index += 1
    #             remain_images -= 1
    #             ed = ed_image
    #         else:
    #             t, h, w = (
    #                 video_grid_thw[video_index][0],
    #                 video_grid_thw[video_index][1],
    #                 video_grid_thw[video_index][2],
    #             )
    #             video_index += 1
    #             remain_frames -= 1
    #             ed = ed_video

    #         llm_grid_t, llm_grid_h, llm_grid_w = (
    #             t,
    #             h // spatial_merge_size,
    #             w // spatial_merge_size,
    #         )
    #         text_len = ed - st

    #         st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
    #         llm_pos_ids_list.append(
    #             torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
    #         )
    #         # ---------------------------------------------------------------- #
    #         t_index = (
    #             torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w)
    #         ).flatten()
    #         # ---------------------------------------------------------------- #

    #         h_index = (
    #             torch.arange(llm_grid_h)
    #             .view(1, -1, 1)
    #             .expand(llm_grid_t, -1, llm_grid_w)
    #             .flatten()
    #         )
    #         w_index = (
    #             torch.arange(llm_grid_w)
    #             .view(1, 1, -1)
    #             .expand(llm_grid_t, llm_grid_h, -1)
    #             .flatten()
    #         )
    #         llm_pos_ids_list.append(
    #             torch.stack([t_index, h_index, w_index]) + text_len + st_idx
    #         )
    #         st = ed + llm_grid_t * llm_grid_h * llm_grid_w

    #     if st < len(input_tokens):
    #         st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
    #         text_len = len(input_tokens) - st
    #         llm_pos_ids_list.append(
    #             torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
    #         )

    #     llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
    #     mrope_position_delta = (llm_positions.max() + 1 - len(input_tokens)).item()
    #     llm_positions = llm_positions[:, context_len:seq_len]

    #     return llm_positions, mrope_position_delta

    # @classmethod
    # def _vl_get_input_positions_tensor(
    #     cls,
    #     input_tokens: list[int],
    #     hf_config: PretrainedConfig,
    #     image_grid_thw: Union[list[list[int]], torch.Tensor],
    #     video_grid_thw: Union[list[list[int]], torch.Tensor],
    #     second_per_grid_ts: list[float],
    #     context_len: int = 0,
    #     seq_len: Optional[int] = None,
    # ) -> tuple[torch.Tensor, int]:
    #     """Get mrope input positions and delta value."""

    #     image_token_id = hf_config.image_token_id
    #     video_token_id = hf_config.video_token_id
    #     vision_start_token_id = hf_config.vision_start_token_id
    #     spatial_merge_size = hf_config.vision_config.spatial_merge_size
    #     tokens_per_second = getattr(hf_config.vision_config, "tokens_per_second", 1.0)

    #     input_tokens_tensor = torch.tensor(input_tokens)
    #     vision_start_indices = torch.argwhere(
    #         input_tokens_tensor == vision_start_token_id
    #     ).squeeze(1)
    #     vision_tokens = input_tokens_tensor[vision_start_indices + 1]
    #     image_nums = (vision_tokens == image_token_id).sum()
    #     video_nums = (vision_tokens == video_token_id).sum()
    #     llm_pos_ids_list: list = []

    #     st = 0
    #     remain_images, remain_videos = image_nums, video_nums
    #     image_index, video_index = 0, 0
    #     for _ in range(image_nums + video_nums):
    #         video_second_per_grid_t = 0.0
    #         if remain_images > 0:
    #             try:
    #                 ed_image = input_tokens.index(image_token_id, st)
    #             except ValueError:
    #                 ed_image = len(input_tokens) + 1
    #         else:
    #             ed_image = len(input_tokens) + 1
    #         if remain_videos > 0:
    #             try:
    #                 ed_video = input_tokens.index(video_token_id, st)
    #             except ValueError:
    #                 ed_video = len(input_tokens) + 1
    #         else:
    #             ed_video = len(input_tokens) + 1
    #         if ed_image < ed_video:
    #             t, h, w = (
    #                 image_grid_thw[image_index][0],
    #                 image_grid_thw[image_index][1],
    #                 image_grid_thw[image_index][2],
    #             )
    #             image_index += 1
    #             remain_images -= 1
    #             ed = ed_image
    #         else:
    #             t, h, w = (
    #                 video_grid_thw[video_index][0],
    #                 video_grid_thw[video_index][1],
    #                 video_grid_thw[video_index][2],
    #             )
    #             video_second_per_grid_t = 1.0
    #             if second_per_grid_ts:
    #                 video_second_per_grid_t = second_per_grid_ts[video_index]
    #             video_index += 1
    #             remain_videos -= 1
    #             ed = ed_video

    #         llm_grid_t, llm_grid_h, llm_grid_w = (
    #             t,
    #             h // spatial_merge_size,
    #             w // spatial_merge_size,
    #         )
    #         text_len = ed - st

    #         st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
    #         llm_pos_ids_list.append(
    #             torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
    #         )
    #         # ---------------------------------------------------------------- #
    #         t_index = (
    #             torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w)
    #             * video_second_per_grid_t
    #             * tokens_per_second
    #         ).flatten()
    #         # ---------------------------------------------------------------- #

    #         h_index = (
    #             torch.arange(llm_grid_h)
    #             .view(1, -1, 1)
    #             .expand(llm_grid_t, -1, llm_grid_w)
    #             .flatten()
    #         )
    #         w_index = (
    #             torch.arange(llm_grid_w)
    #             .view(1, 1, -1)
    #             .expand(llm_grid_t, llm_grid_h, -1)
    #             .flatten()
    #         )
    #         llm_pos_ids_list.append(
    #             torch.stack([t_index, h_index, w_index]) + text_len + st_idx
    #         )
    #         st = ed + llm_grid_t * llm_grid_h * llm_grid_w

    #     if st < len(input_tokens):
    #         st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
    #         text_len = len(input_tokens) - st
    #         llm_pos_ids_list.append(
    #             torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
    #         )

    #     llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
    #     mrope_position_delta = (llm_positions.max() + 1 - len(input_tokens)).item()
    #     llm_positions = llm_positions[:, context_len:seq_len]

    #     return llm_positions, mrope_position_delta

    # @classmethod
    # def _omni3_get_input_positions_tensor(
    #     cls,
    #     config,
    #     input_ids: torch.Tensor,
    #     image_grid_thw: torch.Tensor,
    #     video_grid_thw: torch.Tensor,
    #     use_audio_in_video: bool = False,
    #     audio_seqlens: Optional[torch.Tensor] = None,
    #     second_per_grids: Optional[torch.Tensor] = None,
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     def _get_feat_extract_output_lengths(input_lengths: torch.LongTensor):
    #         input_lengths_leave = input_lengths % 100
    #         feat_lengths = (input_lengths_leave - 1) // 2 + 1
    #         output_lengths = (
    #             ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    #         )
    #         return output_lengths

    #     if input_ids is None or input_ids.ndim != 1:
    #         raise ValueError("_omni3_get_input_positions_tensor expects 1D input_ids")

    #     seq_len = input_ids.shape[0]
    #     device = input_ids.device
    #     dtype = input_ids.dtype

    #     if image_grid_thw is not None:
    #         image_grid_thw = image_grid_thw.to(device=device, dtype=torch.long)
    #     if video_grid_thw is not None:
    #         video_grid_thw = video_grid_thw.to(device=device, dtype=torch.long)

    #     if second_per_grids is None:
    #         if video_grid_thw is not None and video_grid_thw.numel() > 0:
    #             second_per_grids = torch.ones(
    #                 video_grid_thw.shape[0], dtype=torch.float32, device=device
    #             )
    #         else:
    #             second_per_grids = torch.tensor([], dtype=torch.float32, device=device)
    #     else:
    #         second_per_grids = second_per_grids.to(device=device, dtype=torch.float32)

    #     if audio_seqlens is not None:
    #         audio_seqlens = audio_seqlens.to(device=device, dtype=torch.long)

    #     spatial_merge_size = config.vision_config.spatial_merge_size
    #     image_token_id = config.image_token_id
    #     video_token_id = config.video_token_id
    #     audio_token_id = config.audio_token_id
    #     vision_start_token_id = config.vision_start_token_id
    #     audio_start_token_id = config.audio_start_token_id
    #     position_id_per_seconds = config.position_id_per_seconds

    #     vision_start_indices = torch.argwhere(
    #         input_ids == vision_start_token_id
    #     ).squeeze(1)
    #     if vision_start_indices.numel() > 0:
    #         vision_tokens = input_ids[vision_start_indices + 1]
    #     else:
    #         vision_tokens = input_ids.new_empty((0,), dtype=input_ids.dtype)
    #     audio_nums = torch.sum(input_ids == audio_start_token_id)
    #     image_nums = (vision_tokens == image_token_id).sum()
    #     video_nums = (
    #         (vision_tokens == audio_start_token_id).sum()
    #         if use_audio_in_video
    #         else (vision_tokens == video_token_id).sum()
    #     )

    #     input_tokens = input_ids.tolist()
    #     llm_pos_ids_list: list[torch.Tensor] = []
    #     st = 0
    #     image_idx = 0
    #     video_idx = 0
    #     audio_idx = 0
    #     remain_images, remain_videos, remain_audios = image_nums, video_nums, audio_nums  # noqa: E501
    #     multimodal_nums = (
    #         image_nums + audio_nums
    #         if use_audio_in_video
    #         else image_nums + video_nums + audio_nums
    #     )  # noqa: E501

    #     for _ in range(multimodal_nums):
    #         st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
    #         if (image_token_id in input_tokens or video_token_id in input_tokens) and (
    #             remain_videos > 0 or remain_images > 0
    #         ):
    #             ed_vision_start = input_tokens.index(vision_start_token_id, st)
    #         else:
    #             ed_vision_start = len(input_tokens) + 1
    #         if audio_token_id in input_tokens and remain_audios > 0:
    #             ed_audio_start = input_tokens.index(audio_start_token_id, st)
    #         else:
    #             ed_audio_start = len(input_tokens) + 1
    #         min_ed = min(ed_vision_start, ed_audio_start)

    #         if min_ed == ed_audio_start:
    #             text_len = min_ed - st
    #             if text_len != 0:
    #                 st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
    #                 llm_pos_ids_list.append(
    #                     torch.arange(text_len, device=device, dtype=torch.long)
    #                     .view(1, -1)
    #                     .expand(3, -1)
    #                     + st_idx
    #                 )
    #             st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
    #             bos_len = 1
    #             llm_pos_ids_list.append(
    #                 torch.arange(bos_len, device=device, dtype=torch.long)
    #                 .view(1, -1)
    #                 .expand(3, -1)
    #                 + st_idx
    #             )
    #             st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
    #             audio_len = _get_feat_extract_output_lengths(audio_seqlens[audio_idx])
    #             llm_pos_ids = (
    #                 torch.arange(audio_len, device=device, dtype=torch.long)
    #                 .view(1, -1)
    #                 .expand(3, -1)
    #                 + st_idx
    #             )
    #             llm_pos_ids_list.append(llm_pos_ids)
    #             st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
    #             eos_len = 1
    #             llm_pos_ids_list.append(
    #                 torch.arange(eos_len, device=device, dtype=torch.long)
    #                 .view(1, -1)
    #                 .expand(3, -1)
    #                 + st_idx
    #             )
    #             st += text_len + bos_len + audio_len + eos_len
    #             audio_idx += 1
    #             remain_audios -= 1
    #         elif (
    #             min_ed == ed_vision_start
    #             and input_ids[ed_vision_start + 1] == image_token_id
    #         ):
    #             text_len = min_ed - st
    #             if text_len != 0:
    #                 st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
    #                 llm_pos_ids_list.append(
    #                     torch.arange(text_len, device=device, dtype=torch.long)
    #                     .view(1, -1)
    #                     .expand(3, -1)
    #                     + st_idx
    #                 )
    #             st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
    #             bos_len = 1
    #             llm_pos_ids_list.append(
    #                 torch.arange(bos_len, device=device, dtype=torch.long)
    #                 .view(1, -1)
    #                 .expand(3, -1)
    #                 + st_idx
    #             )
    #             st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
    #             grid_t = image_grid_thw[image_idx][0]
    #             grid_hs = image_grid_thw[:, 1]
    #             grid_ws = image_grid_thw[:, 2]
    #             t_index = torch.arange(grid_t, device=device) * position_id_per_seconds
    #             llm_pos_ids = cls._get_llm_pos_ids_for_vision(
    #                 st_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws
    #             )
    #             image_len = image_grid_thw[image_idx].prod() // (spatial_merge_size**2)
    #             llm_pos_ids_list.append(llm_pos_ids)
    #             st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
    #             eos_len = 1
    #             llm_pos_ids_list.append(
    #                 torch.arange(eos_len, device=device, dtype=torch.long)
    #                 .view(1, -1)
    #                 .expand(3, -1)
    #                 + st_idx
    #             )
    #             st += text_len + bos_len + image_len + eos_len
    #             image_idx += 1
    #             remain_images -= 1
    #         elif (
    #             min_ed == ed_vision_start
    #             and input_ids[ed_vision_start + 1] == video_token_id
    #             and not use_audio_in_video
    #         ):
    #             text_len = min_ed - st
    #             if text_len != 0:
    #                 st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
    #                 llm_pos_ids_list.append(
    #                     torch.arange(text_len, device=device, dtype=torch.long)
    #                     .view(1, -1)
    #                     .expand(3, -1)
    #                     + st_idx
    #                 )
    #             st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
    #             bos_len = 1
    #             llm_pos_ids_list.append(
    #                 torch.arange(bos_len, device=device, dtype=torch.long)
    #                 .view(1, -1)
    #                 .expand(3, -1)
    #                 + st_idx
    #             )
    #             st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
    #             grid_t = video_grid_thw[video_idx][0]
    #             grid_hs = video_grid_thw[:, 1]
    #             grid_ws = video_grid_thw[:, 2]
    #             t_index = (
    #                 torch.arange(grid_t, device=device)
    #                 * float(second_per_grids[video_idx].item())
    #                 * position_id_per_seconds
    #             )
    #             llm_pos_ids = cls._get_llm_pos_ids_for_vision(
    #                 st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
    #             )
    #             video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
    #             llm_pos_ids_list.append(llm_pos_ids)
    #             st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
    #             eos_len = 1
    #             llm_pos_ids_list.append(
    #                 torch.arange(eos_len, device=device, dtype=torch.long)
    #                 .view(1, -1)
    #                 .expand(3, -1)
    #                 + st_idx
    #             )
    #             st += text_len + bos_len + video_len + eos_len
    #             video_idx += 1
    #             remain_videos -= 1
    #         elif (
    #             min_ed == ed_vision_start
    #             and ed_vision_start + 1 == ed_audio_start
    #             and use_audio_in_video
    #         ):
    #             text_len = min_ed - st
    #             if text_len != 0:
    #                 st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
    #                 llm_pos_ids_list.append(
    #                     torch.arange(text_len, device=device, dtype=torch.long)
    #                     .view(1, -1)
    #                     .expand(3, -1)
    #                     + st_idx
    #                 )
    #             st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
    #             bos_len = 1
    #             bos_block = (
    #                 torch.arange(bos_len, device=device, dtype=torch.long)
    #                 .view(1, -1)
    #                 .expand(3, -1)
    #                 + st_idx
    #             )
    #             llm_pos_ids_list.append(bos_block)
    #             llm_pos_ids_list.append(bos_block)
    #             st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
    #             audio_len = _get_feat_extract_output_lengths(audio_seqlens[audio_idx])
    #             audio_llm_pos_ids = (
    #                 torch.arange(audio_len, device=device, dtype=torch.long)
    #                 .view(1, -1)
    #                 .expand(3, -1)
    #                 + st_idx
    #             )
    #             grid_t = video_grid_thw[video_idx][0]
    #             grid_hs = video_grid_thw[:, 1]
    #             grid_ws = video_grid_thw[:, 2]
    #             t_index = (
    #                 torch.arange(grid_t, device=device)
    #                 * float(second_per_grids[video_idx].item())
    #                 * position_id_per_seconds
    #             )
    #             video_llm_pos_ids = cls._get_llm_pos_ids_for_vision(
    #                 st_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
    #             )
    #             video_data_index, audio_data_index = 0, 0
    #             while (
    #                 video_data_index < video_llm_pos_ids.shape[-1]
    #                 and audio_data_index < audio_llm_pos_ids.shape[-1]
    #             ):
    #                 if (
    #                     video_llm_pos_ids[0][video_data_index]
    #                     <= audio_llm_pos_ids[0][audio_data_index]
    #                 ):
    #                     llm_pos_ids_list.append(
    #                         video_llm_pos_ids[
    #                             :, video_data_index : video_data_index + 1
    #                         ]
    #                     )
    #                     video_data_index += 1
    #                 else:
    #                     llm_pos_ids_list.append(
    #                         audio_llm_pos_ids[
    #                             :, audio_data_index : audio_data_index + 1
    #                         ]
    #                     )
    #                     audio_data_index += 1
    #             if video_data_index < video_llm_pos_ids.shape[-1]:
    #                 llm_pos_ids_list.append(
    #                     video_llm_pos_ids[
    #                         :, video_data_index : video_llm_pos_ids.shape[-1]
    #                     ]
    #                 )
    #             if audio_data_index < audio_llm_pos_ids.shape[-1]:
    #                 llm_pos_ids_list.append(
    #                     audio_llm_pos_ids[
    #                         :, audio_data_index : audio_llm_pos_ids.shape[-1]
    #                     ]
    #                 )
    #             video_len = video_grid_thw[video_idx].prod() // (spatial_merge_size**2)
    #             st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
    #             eos_len = 1
    #             eos_block = (
    #                 torch.arange(eos_len, device=device, dtype=torch.long)
    #                 .view(1, -1)
    #                 .expand(3, -1)
    #                 + st_idx
    #             )
    #             llm_pos_ids_list.append(eos_block)
    #             llm_pos_ids_list.append(eos_block)
    #             st += text_len + bos_len * 2 + audio_len + video_len + eos_len * 2  # noqa: E501
    #             audio_idx += 1
    #             video_idx += 1
    #             remain_videos -= 1
    #             remain_audios -= 1

    #     if st < len(input_tokens):
    #         st_idx = llm_pos_ids_list[-1].max() + 1 if llm_pos_ids_list else 0
    #         text_len = len(input_tokens) - st
    #         llm_pos_ids_list.append(
    #             torch.arange(text_len, device=device, dtype=torch.long)
    #             .view(1, -1)
    #             .expand(3, -1)
    #             + st_idx
    #         )

    #     llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
    #     if llm_positions.shape[1] != seq_len:
    #         raise RuntimeError("Position ids length mismatch with input ids length")

    #     position_ids = llm_positions.to(device=device, dtype=dtype)
    #     mrope_position_delta = llm_positions.max() + 1 - seq_len
    #     return position_ids, mrope_position_delta

    # @classmethod
    # def _omni_get_input_positions_tensor(
    #     cls,
    #     input_tokens: list[int],
    #     hf_config: PretrainedConfig,
    #     image_grid_thw: Union[list[list[int]], torch.Tensor],
    #     video_grid_thw: Union[list[list[int]], torch.Tensor],
    #     second_per_grid_ts: Optional[list[float]] = None,
    #     context_len: int = 0,
    #     seq_len: Optional[int] = None,
    #     audio_feature_lengths: Optional[torch.Tensor] = None,
    #     use_audio_in_video: bool = False,
    # ) -> tuple[torch.Tensor, int]:
    #     """Get mrope input positions and delta value (Qwen2.5-Omni version).

    #     Differences from MRotaryEmbedding:
    #         1. Add audio support (and related `audio_feature_lengths`).
    #         2. Add `use_audio_in_video` option to read audio from video inputs.
    #             In this case, audio and vision position ids will be split into
    #             chunks and interleaved.

    #     Example:

    #         (V_i are vision position ids, A_i are audio position ids)

    #         |V_1 ...    V_n|A_1 ...   A_n|V_n+1 ... V_2n|A_n+1 ... A_2n|...
    #         |vision chunk 1|audio chunk 1|vision chunk 2|audio chunk 2 |...
    #     """

    #     # TODO(fyabc): refactor and share more code with
    #     #  _vl_get_input_positions_tensor.

    #     # ---------------------------------------------------------------- #
    #     model_type = hf_config.model_type
    #     # ---------------------------------------------------------------- #

    #     thinker_config = hf_config.thinker_config

    #     if isinstance(image_grid_thw, list):
    #         image_grid_thw = torch.tensor(image_grid_thw)
    #     if isinstance(video_grid_thw, list):
    #         video_grid_thw = torch.tensor(video_grid_thw)

    #     # ---------------------------------------------------------------- #
    #     if "qwen3_omni" in model_type:
    #         input_tensor = torch.tensor(input_tokens)
    #         audio_lengths_tensor = audio_feature_lengths
    #         if audio_lengths_tensor is not None and not isinstance(
    #             audio_lengths_tensor, torch.Tensor
    #         ):
    #             audio_lengths_tensor = torch.as_tensor(
    #                 audio_lengths_tensor, dtype=torch.long
    #             )
    #         second_per_grids_tensor = (
    #             torch.tensor(second_per_grid_ts) if second_per_grid_ts else None
    #         )

    #         llm_positions, mrope_position_delta = cls._omni3_get_input_positions_tensor(  # noqa: E501
    #             thinker_config,
    #             input_tensor,
    #             image_grid_thw,
    #             video_grid_thw,
    #             use_audio_in_video,
    #             audio_lengths_tensor,
    #             second_per_grids_tensor,
    #         )
    #         return llm_positions, mrope_position_delta
    #         # ---------------------------------------------------------------- #

    #     audio_token_id = thinker_config.audio_token_index
    #     image_token_id = thinker_config.image_token_index
    #     video_token_id = thinker_config.video_token_index
    #     audio_start_token_id = thinker_config.audio_start_token_id
    #     audio_end_token_id = thinker_config.audio_end_token_id
    #     vision_start_token_id = thinker_config.vision_start_token_id
    #     vision_end_token_id = thinker_config.vision_end_token_id
    #     seconds_per_chunk = thinker_config.seconds_per_chunk
    #     spatial_merge_size = thinker_config.vision_config.spatial_merge_size
    #     tokens_per_second = getattr(
    #         thinker_config.vision_config, "tokens_per_second", 25
    #     )

    #     src_item = input_tokens
    #     audio_seqlens = audio_feature_lengths
    #     if not second_per_grid_ts:
    #         second_per_grid_ts = [1] * video_grid_thw.shape[0]
    #     audio_idx = 0
    #     video_idx = 0
    #     image_idx = 0
    #     new_src_item: list[int] = []
    #     llm_pos_ids_list: list[torch.Tensor] = []

    #     idx = 0
    #     while idx < len(src_item):
    #         new_src_item_len = len(new_src_item)
    #         start_idx = (
    #             llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
    #         )
    #         if src_item[idx] not in [audio_token_id, video_token_id, image_token_id]:
    #             if use_audio_in_video and idx > 0:
    #                 if (
    #                     src_item[idx] == vision_end_token_id
    #                     and src_item[idx - 1] == audio_end_token_id
    #                 ):
    #                     # processing the <|audio_eos|> before <|vision_eos|>
    #                     start_idx -= 1
    #                 elif (
    #                     src_item[idx] == audio_start_token_id
    #                     and src_item[idx - 1] == vision_start_token_id
    #                 ):
    #                     # processing the <|audio_bos|> after <|vision_eos|>
    #                     start_idx -= 1
    #             new_src_item.append(src_item[idx])
    #             llm_pos_ids = torch.tensor([start_idx], dtype=torch.long).expand(3, -1)
    #             llm_pos_ids_list.append(llm_pos_ids)
    #         elif src_item[idx] == audio_token_id:
    #             assert audio_seqlens is not None
    #             audio_seqlen = audio_seqlens[audio_idx]
    #             place_num = ((audio_seqlen - 1) // 2 + 1 - 2) // 2 + 1
    #             new_src_item.extend([audio_token_id] * place_num)
    #             llm_pos_ids = torch.arange(place_num).expand(3, -1) + start_idx
    #             llm_pos_ids_list.append(llm_pos_ids)
    #             audio_idx += 1
    #         elif src_item[idx] == image_token_id:
    #             grid_t = image_grid_thw[image_idx][0]
    #             grid_hs = image_grid_thw[:, 1]
    #             grid_ws = image_grid_thw[:, 2]
    #             # ---------------------------------------------------------------- #
    #             t_index = torch.arange(grid_t) * 1 * tokens_per_second
    #             # ---------------------------------------------------------------- #
    #             llm_pos_ids = cls._get_llm_pos_ids_for_vision(
    #                 start_idx, image_idx, spatial_merge_size, t_index, grid_hs, grid_ws
    #             )
    #             llm_pos_ids_list.append(llm_pos_ids)
    #             vision_seqlen = image_grid_thw[image_idx].prod() // (
    #                 spatial_merge_size**2
    #             )
    #             new_src_item.extend([image_token_id] * vision_seqlen)
    #             image_idx += 1
    #         elif src_item[idx] == video_token_id and not use_audio_in_video:
    #             grid_t = video_grid_thw[video_idx][0]
    #             grid_hs = video_grid_thw[:, 1]
    #             grid_ws = video_grid_thw[:, 2]
    #             # ---------------------------------------------------------------- #
    #             t_index = (
    #                 torch.arange(grid_t)
    #                 * second_per_grid_ts[video_idx]
    #                 * tokens_per_second
    #             )
    #             # ---------------------------------------------------------------- #
    #             llm_pos_ids = cls._get_llm_pos_ids_for_vision(
    #                 start_idx, video_idx, spatial_merge_size, t_index, grid_hs, grid_ws
    #             )
    #             llm_pos_ids_list.append(llm_pos_ids)
    #             vision_seqlen = video_grid_thw[video_idx].prod() // (
    #                 spatial_merge_size**2
    #             )
    #             new_src_item.extend([video_token_id] * vision_seqlen)
    #             video_idx += 1
    #         else:
    #             # read audio from video
    #             assert audio_seqlens is not None
    #             audio_seqlen = audio_seqlens[audio_idx]
    #             vision_seqlen = video_grid_thw[video_idx].prod() // (
    #                 spatial_merge_size**2
    #             )
    #             grid_t = video_grid_thw[video_idx][0]
    #             grid_h = video_grid_thw[video_idx][1]
    #             grid_w = video_grid_thw[video_idx][2]
    #             grid_hs = video_grid_thw[:, 1]
    #             grid_ws = video_grid_thw[:, 2]
    #             t_ntoken_per_chunk = int(tokens_per_second * seconds_per_chunk)
    #             t_index = (
    #                 torch.arange(grid_t)
    #                 * second_per_grid_ts[video_idx]
    #                 * tokens_per_second
    #             )
    #             t_index_split_chunk = cls._split_list_into_ranges(
    #                 t_index, t_ntoken_per_chunk
    #             )
    #             place_num = (((audio_seqlen - 1) // 2 + 1 - 2) // 2 + 1) + 2
    #             pure_audio_len = place_num - 2
    #             added_audio_len = 0
    #             audio_llm_pos_ids_list: list[torch.Tensor] = []
    #             for t_chunk in t_index_split_chunk:
    #                 vision_ntoken_per_chunk = (
    #                     len(t_chunk) * grid_h * grid_w // (spatial_merge_size**2)
    #                 )
    #                 new_src_item.extend([video_token_id] * vision_ntoken_per_chunk)
    #                 vision_llm_pos_ids_list = cls._get_llm_pos_ids_for_vision(
    #                     start_idx,
    #                     video_idx,
    #                     spatial_merge_size,
    #                     t_chunk,
    #                     grid_hs,
    #                     grid_ws,
    #                 ).split(1, dim=1)
    #                 llm_pos_ids_list.extend(vision_llm_pos_ids_list)
    #                 new_src_item.extend(
    #                     min(t_ntoken_per_chunk, pure_audio_len - added_audio_len)
    #                     * [audio_token_id]
    #                 )
    #                 audio_start_idx = (
    #                     start_idx
    #                     if len(audio_llm_pos_ids_list) == 0
    #                     else audio_llm_pos_ids_list[-1][0].item() + 1
    #                 )
    #                 if min(t_ntoken_per_chunk, pure_audio_len - added_audio_len) > 0:
    #                     audio_llm_pos_ids_list = (
    #                         torch.arange(
    #                             min(
    #                                 t_ntoken_per_chunk, pure_audio_len - added_audio_len
    #                             )
    #                         ).expand(3, -1)
    #                         + audio_start_idx
    #                     ).split(1, dim=1)
    #                 else:
    #                     audio_llm_pos_ids_list = []
    #                 added_audio_len += min(
    #                     t_ntoken_per_chunk, pure_audio_len - added_audio_len
    #                 )
    #                 llm_pos_ids_list.extend(audio_llm_pos_ids_list)
    #             if added_audio_len < pure_audio_len:
    #                 new_src_item.extend(
    #                     (pure_audio_len - added_audio_len) * [audio_token_id]
    #                 )
    #                 audio_llm_pos_ids_list = (
    #                     torch.arange(pure_audio_len - added_audio_len).expand(3, -1)
    #                     + llm_pos_ids_list[-1].max()
    #                     + 1
    #                 ).split(1, dim=1)
    #                 llm_pos_ids_list.extend(audio_llm_pos_ids_list)
    #             audio_idx += 1
    #             video_idx += 1
    #         # move to the next token
    #         idx += len(new_src_item) - new_src_item_len

    #     llm_positions = torch.cat(llm_pos_ids_list, dim=1)
    #     mrope_position_delta = (
    #         torch.cat(llm_pos_ids_list, dim=1).max() + 1 - len(src_item)
    #     )
    #     llm_positions = llm_positions[:, context_len:seq_len]

    #     return llm_positions, mrope_position_delta

    # @classmethod
    # def omni_get_updates_use_audio_in_video(
    #     cls,
    #     thinker_config: PretrainedConfig,
    #     audio_len: int,
    #     video_grid_thw: Union[list[int], torch.Tensor],
    #     video_second_per_grid_t: float,
    # ) -> list[int]:
    #     """Get video prompt updates when `use_audio_in_video` is True.

    #     In this case, audio and vision update ids will be split into
    #     chunks and interleaved (details in `_omni_get_input_positions_tensor`).

    #     <|video_bos|><|VIDEO|><|video_eos|> =>
    #     <|video_bos|><|audio_bos|>(... chunks ...)<|audio_eos|><|video_eos|>
    #     """

    #     audio_token_id = thinker_config.audio_token_index
    #     video_token_id = thinker_config.video_token_index
    #     audio_start_token_id = thinker_config.audio_start_token_id
    #     audio_end_token_id = thinker_config.audio_end_token_id
    #     seconds_per_chunk = thinker_config.seconds_per_chunk
    #     spatial_merge_size = thinker_config.vision_config.spatial_merge_size
    #     tokens_per_second = getattr(
    #         thinker_config.vision_config, "tokens_per_second", 25
    #     )

    #     grid_t = video_grid_thw[0]
    #     grid_h = video_grid_thw[1]
    #     grid_w = video_grid_thw[2]
    #     t_ntoken_per_chunk = int(tokens_per_second * seconds_per_chunk)
    #     # ---------------------------------------------------------------- #
    #     t_index = torch.arange(grid_t) * video_second_per_grid_t * tokens_per_second
    #     # ---------------------------------------------------------------- #

    #     t_index_split_chunk = cls._split_list_into_ranges(t_index, t_ntoken_per_chunk)

    #     updates = [audio_start_token_id]
    #     added_audio_len = 0
    #     for t_chunk in t_index_split_chunk:
    #         vision_ntoken_per_chunk = (
    #             len(t_chunk) * grid_h * grid_w // (spatial_merge_size**2)
    #         )
    #         updates.extend([video_token_id] * vision_ntoken_per_chunk)

    #         audio_chunk_size = min(t_ntoken_per_chunk, audio_len - added_audio_len)
    #         updates.extend(audio_chunk_size * [audio_token_id])
    #         added_audio_len += audio_chunk_size
    #     if added_audio_len < audio_len:
    #         updates.extend((audio_len - added_audio_len) * [audio_token_id])
    #     updates.extend([audio_end_token_id])

    #     return updates


import vllm.model_executor.layers.rotary_embedding.mrope

vllm.model_executor.layers.rotary_embedding.mrope.MRotaryEmbedding.get_input_positions_tensor = MacaMRotaryEmbedding.get_input_positions_tensor
