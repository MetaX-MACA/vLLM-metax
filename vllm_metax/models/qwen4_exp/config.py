# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Normalize the FlexSmoothQuant QSA alias before Transformers validation."""

from vllm.models.qwen4_exp.config import (
    Qwen4ExpConfig as UpstreamQwen4ExpConfig,
    Qwen4ExpTextConfig as UpstreamQwen4ExpTextConfig,
)


class Qwen4ExpTextConfig(UpstreamQwen4ExpTextConfig):
    def __init__(self, *args, layer_types=None, **kwargs):
        if layer_types is not None:
            layer_types = [
                "full_attention" if kind == "qwen_sparse_attention" else kind
                for kind in layer_types
            ]
        super().__init__(*args, layer_types=layer_types, **kwargs)


class Qwen4ExpConfig(UpstreamQwen4ExpConfig):
    sub_configs = {
        **UpstreamQwen4ExpConfig.sub_configs,
        "text_config": Qwen4ExpTextConfig,
    }

    def __init__(self, *args, **kwargs):
        # Preserve the upstream nested-config constructor under Transformers 5.19.
        super().__init__(*args, **kwargs)
