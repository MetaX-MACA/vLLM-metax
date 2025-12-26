# SPDX-License-Identifier: Apache-2.0
from vllm.attention.layers.mm_encoder_attention import MMEncoderAttention, logger
import torch
from vllm.config import MultiModalConfig

from vllm_metax.attention.ops.vit_attn_wrappers import vit_flash_attn_wrapper
from vllm.attention.backends.registry import AttentionBackendEnum
from vllm.model_executor.models.vision import get_vit_attn_backend


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
        super(MMEncoderAttention, self).__init__()

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.layer_name = prefix

        assert self.num_heads % self.num_kv_heads == 0, (
            f"num_heads ({self.num_heads}) is not "
            f"divisible by num_kv_heads ({self.num_kv_heads})"
        )
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        # During model initialization, the default dtype is set as the model
        # weight and activation dtype.
        dtype = torch.get_default_dtype()

        # Try to get vision attention backend from multimodal_config.
        attn_backend_override = None
        if multimodal_config is not None:
            attn_backend_override = multimodal_config.mm_encoder_attn_backend

        # Get device-specific vision attention backend.
        self.attn_backend = get_vit_attn_backend(
            head_size=head_size,
            dtype=dtype,
            attn_backend_override=attn_backend_override,
        )

        self.is_flash_attn_backend = self.attn_backend in {
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.ROCM_AITER_FA,
        }

        # /------------------ Metax Modification -------------------\
        from vllm_metax.attention.utils.fa_utils import flash_attn_varlen_func
        # \---------------------------------------------------------/

        self.flash_attn_varlen_func = flash_attn_varlen_func

        logger.info_once(f"Using {self.attn_backend} for MMEncoderAttention.")

    def _forward_fa(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: torch.Tensor | None = None,  # Only used for Flash Attention
    ) -> torch.Tensor:
        assert self.flash_attn_varlen_func is not None, (
            "Flash attention function is not set."
        )
        # # TODO(Isotr0py): Migrate MultiHeadAttention
        assert cu_seqlens is not None and max_seqlen is not None

        bsz = query.shape[0]

        output = vit_flash_attn_wrapper(
            q=query,
            k=key,
            v=value,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            batch_size=bsz,
            is_rocm_aiter=False,
        )
        return output

    def forward_oop(self, *args, **kwargs):
        # Custom forward method for MACA can be implemented here.
        return self.forward_cuda(self, *args, **kwargs)
