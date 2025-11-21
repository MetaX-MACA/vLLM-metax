# SPDX-License-Identifier: Apache-2.0
# 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved. 
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding


@RotaryEmbedding.register_oot
class MacaRotaryEmbedding(RotaryEmbedding):

    def forward_oot(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)
