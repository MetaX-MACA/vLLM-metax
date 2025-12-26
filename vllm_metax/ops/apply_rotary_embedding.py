# SPDX-License-Identifier: Apache-2.0
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
import torch


class MacaApplyRotaryEmb(ApplyRotaryEmb):
    def forward_oot(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        from flash_attn.layers.rotary import apply_rotary_emb

        origin_dtype = x.dtype
        if self.enable_fp32_compute:
            x = x.float()
            cos = cos.float()
            sin = sin.float()

        origin_shape = x.shape
        if len(origin_shape) == 3:
            # x: [seq_len, num_heads, head_size]
            x = x.unsqueeze(0)

        """
        Arguments of apply_rotary_emb() in vllm_flash_attn:
            x: [batch_size, seq_len, nheads, headdim]
            cos, sin: [seqlen_rotary, rotary_dim / 2]
            interleaved: defalut as False (Neox-style).
            ...
        """
        interleaved = not self.is_neox_style
        output = apply_rotary_emb(x, cos, sin, interleaved)

        if len(origin_shape) == 3:
            output = output.squeeze(0)
        if self.enable_fp32_compute:
            output = output.to(origin_dtype)
        return output
