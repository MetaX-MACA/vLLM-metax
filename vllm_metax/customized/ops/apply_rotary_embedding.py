# SPDX-License-Identifier: Apache-2.0
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
import torch


@ApplyRotaryEmb.register_oot
class MacaApplyRotaryEmb(ApplyRotaryEmb):
    def forward_oot(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        from flash_attn.layers.rotary import apply_rotary_emb

        x, cos, sin, origin_shape, origin_dtype = self._pre_process(x, cos, sin)

        """
        Arguments of apply_rotary_emb() in vllm_flash_attn:
            x: [batch_size, seq_len, nheads, headdim]
            cos, sin: [seqlen_rotary, rotary_dim / 2]
            interleaved: defalut as False (Neox-style).
            ...
        """
        interleaved = not self.is_neox_style
        output = apply_rotary_emb(x, cos, sin, interleaved)

        output = self._post_process(output, origin_shape, origin_dtype)
        return output
