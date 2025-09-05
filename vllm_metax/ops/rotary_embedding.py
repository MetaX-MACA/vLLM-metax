# SPDX-License-Identifier: Apache-2.0
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

@RotaryEmbedding.register_oot
class MacaRotaryEmbedding(RotaryEmbedding):
    def forward_oot(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)
