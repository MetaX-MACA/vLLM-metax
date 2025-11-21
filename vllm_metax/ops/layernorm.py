# SPDX-License-Identifier: Apache-2.0
# 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved. 
from vllm.model_executor.layers.layernorm import GemmaRMSNorm, RMSNorm


@RMSNorm.register_oot
class MacaRMSNorm(RMSNorm):

    def forward_oot(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)


@GemmaRMSNorm.register_oot
class MacaGemmaRMSNorm(GemmaRMSNorm):

    def forward_oot(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)
