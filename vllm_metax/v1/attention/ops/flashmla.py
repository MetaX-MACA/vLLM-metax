# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# adapted from: https://github.com/deepseek-ai/FlashMLA/blob/main/flash_mla/flash_mla_interface.py

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


def _is_flashmla_available() -> tuple[bool, str | None]:
    """
    Return: is_supported_flag, unsupported_reason (optional).
    """
    return True, None


def is_flashmla_dense_supported() -> tuple[bool, str | None]:
    """
    Return: is_supported_flag, unsupported_reason (optional).
    """
    is_availble, maybe_reason = _is_flashmla_available()
    if not is_availble:
        return False, maybe_reason
    return True, None


def is_flashmla_sparse_supported() -> tuple[bool, str | None]:
    """
    Return: is_supported_flag, unsupported_reason (optional).
    """
    is_availble, maybe_reason = _is_flashmla_available()
    if not is_availble:
        return False, maybe_reason
    return True, None


def _raise_flashmla_unavailable(*_args, **_kwargs):
    _, reason = _is_flashmla_available()
    raise RuntimeError(reason or "FlashMLA is not available")


if _is_flashmla_available()[0]:
    from flash_mla.flash_mla_interface import (  # noqa: F401
        flash_mla_sparse_fwd,
        flash_mla_with_kvcache,
        get_mla_metadata,
    )
else:
    flash_mla_sparse_fwd = _raise_flashmla_unavailable  # type: ignore[assignment]
    flash_mla_with_kvcache = _raise_flashmla_unavailable  # type: ignore[assignment]
    get_mla_metadata = _raise_flashmla_unavailable  # type: ignore[assignment]


def get_mla_metadata_dense_fp8(
    cache_seqlens: torch.Tensor,
    num_q_tokens_per_head_k: int,
    num_heads_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError(
        "Maca does not support FlashMLA get_mla_metadata_dense_fp8 yet."
    )


def flash_mla_with_kvcache_fp8(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    head_dim_v: int,
    tile_scheduler_metadata: torch.Tensor,
    num_splits: torch.Tensor,
    softmax_scale: float | None = None,
    causal: bool = False,
    descale_q: torch.Tensor | None = None,
    descale_k: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    raise NotImplementedError(
        "Maca does not support FlashMLA flash_mla_with_kvcache_fp8 yet."
    )


#
# TODO: Add fake functions
#
# @register_fake("_flashmla_C::get_mla_metadata")
# def _get_mla_metadata_fake(....) -> Tuple[torch.Tensor, torch.Tensor]:
#     return ....
#
# @register_fake("_flashmla_C::fwd_kvcache_mla")
# def _fwd_kvcache_mla_fake(....) -> Tuple[torch.Tensor, torch.Tensor]:
#     return ....
#


# Metax: torch_ref
def torch_flash_mla_sparse_prefill(
    q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, sm_scale: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import math

    def log2sumexp2(a: torch.Tensor, dim: int) -> torch.Tensor:
        return torch.logsumexp(a * math.log(2), dim=dim) * math.log2(math.e)

    assert len(q.shape) == len(kv.shape) == 3  # b == 1
    s_q, _, d_qk = q.shape
    s_kv, _, _ = kv.shape

    indices = indices[:, 0, :]  # [s_q, topk]
    invalid_indices_mask = (indices < 0) | (indices >= s_kv)
    qs = q[:, :, :].float()  # [s_q, h_q, d_qk]
    kvs = kv[:, 0, :].float()  # [s_kv, d_qk]

    _, topk = indices.shape

    kvs = torch.index_select(
        kvs, 0, indices.masked_fill(invalid_indices_mask, 0).flatten()
    ).view(s_q, topk, d_qk)  # [s_q, topk, d_qk]
    attn_score = qs @ kvs.transpose(1, 2)  # [s_q, h_q, topk]
    attn_score.masked_fill_(invalid_indices_mask.unsqueeze(1), float("-inf"))
    attn_score *= sm_scale * math.log2(math.e)
    max_logits = torch.max(attn_score, dim=-1)[0]  # [s_q, h_q]
    lse = log2sumexp2(attn_score, dim=-1)  # [s_q, h_q]
    attn_score = torch.exp2(attn_score - lse.unsqueeze(-1))  # [s_q, h_q, topk]
    result = attn_score @ kvs[:, :, :512]

    return (result.to(torch.bfloat16), max_logits, lse)
