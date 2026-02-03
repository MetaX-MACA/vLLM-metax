# SPDX-License-Identifier: Apache-2.0
from vllm.model_executor.layers.activation import (
    FastGELU,
    FatreluAndMul,
    GeluAndMul,
    MulAndSilu,
    NewGELU,
    QuickGELU,
    SiluAndMul,
    SwigluOAIAndMul,
)


@FatreluAndMul.register_oot
class MacaFatreluAndMul(FatreluAndMul):
    def forward_oot(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)


@SiluAndMul.register_oot
class MacaSiluAndMul(SiluAndMul):
    def forward_oot(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)


@MulAndSilu.register_oot
class MacaMulAndSilu(MulAndSilu):
    def forward_oot(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)


@GeluAndMul.register_oot
class MacaGeluAndMul(GeluAndMul):
    def forward_oot(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)


@SwigluOAIAndMul.register_oot
class MacaSwigluOAIAndMul(SwigluOAIAndMul):
    def forward_oot(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)


@NewGELU.register_oot
class MacaNewGELU(NewGELU):
    def forward_oot(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)


@FastGELU.register_oot
class MacaFastGELU(FastGELU):
    def forward_oot(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)


@QuickGELU.register_oot
class MacaQuickGELU(QuickGELU):
    def forward_oot(self, *args, **kwargs):
        return self.forward_cuda(*args, **kwargs)
