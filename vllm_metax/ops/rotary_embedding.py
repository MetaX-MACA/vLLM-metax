# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding
from vllm.model_executor.layers.rotary_embedding.xdrope import XDRotaryEmbedding


@RotaryEmbedding.register_oot
class MacaRotaryEmbedding(RotaryEmbedding):
    def forward_oot(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)


@MRotaryEmbedding.register_oot
class MacaMRotaryEmbedding(MRotaryEmbedding):
    def forward_oot(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)


@XDRotaryEmbedding.register_oot
class MacaXDRotaryEmbedding(XDRotaryEmbedding):
    def forward_oot(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)
