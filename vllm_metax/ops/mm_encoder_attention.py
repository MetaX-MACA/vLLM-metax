# SPDX-License-Identifier: Apache-2.0
from vllm.model_executor.layers.attention.mm_encoder_attention import (
    MMEncoderAttention,
)
import torch
from vllm.config import MultiModalConfig
from vllm_metax.v1.attention.ops.vit_attn_wrappers import (
    vit_flash_attn_wrapper as mx_vit_fa_wrapper,
)
from vllm_metax.v1.attention.backends.fa_utils import get_flash_attn_version
from vllm.v1.attention.backends.registry import AttentionBackendEnum


@MMEncoderAttention.register_oot
class MacaMMEncoderAttention(MMEncoderAttention):
    """Multi-headed attention with MACA optimizations."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float | None = None,
        num_kv_heads: int | None = None,
        prefix: str = "",
        multimodal_config: MultiModalConfig | None = None,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            prefix,
            multimodal_config,
        )

        # /------------------ Metax Modification -------------------\
        self._fa_version = (
            get_flash_attn_version() if self.is_flash_attn_backend else None
        )
        # \---------------------------------------------------------/

    def _forward_fa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
    ) -> torch.Tensor:
        """Input shape:
        (batch_size x seq_len x hidden_size) or
        (batch_size x seq_len x num_heads x head_size)
        """
        assert (cu_seqlens is not None and max_seqlen is not None) or (
            cu_seqlens is None and max_seqlen is None
        ), "cu_seqlens and max_seqlen should be both set or both None."

        bsz, q_len = query.size()[:2]
        kv_len = key.size(1)
        is_reshaped = query.dim() != 4

        query, key, value = self.maybe_reshape_qkv_to_4d(
            query, key, value, bsz, q_len, kv_len
        )

        output = mx_vit_fa_wrapper(
            q=query,
            k=key,
            v=value,
            batch_size=bsz,
            is_rocm_aiter=(self.attn_backend == AttentionBackendEnum.ROCM_AITER_FA),
            fa_version=self._fa_version,
            scale=self.scale,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        if is_reshaped:
            output = output.reshape(bsz, q_len, -1)
        return output

    def forward_oot(self, *args, **kwargs):
        # Custom forward method for MACA can be implemented here.
        return self.forward_cuda(*args, **kwargs)
