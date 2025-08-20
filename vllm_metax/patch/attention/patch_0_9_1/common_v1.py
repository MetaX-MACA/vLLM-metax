# SPDX-License-Identifier: Apache-2.0

# TODO: remove this file

import vllm
from vllm_metax.patch.hook_registry import register_patch
from vllm import envs
from vllm.v1.attention.backends.mla.common import logger

import torch
from vllm.model_executor.layers.linear import (LinearBase,
                                               UnquantizedLinearMethod)

from vllm_metax.v1.attention.backends.mla.common import (MetaxMLACommonMetadataBuilder,
                                                                MetaxMLACommonImpl)
from flash_attn import flash_attn_varlen_func
def get_flash_attn_version():
    return None

from vllm.v1.attention.backends.mla import common

vllm.v1.attention.backends.mla.common.is_vllm_fa = False
vllm.v1.attention.backends.mla.common.get_flash_attn_version = get_flash_attn_version


vllm.v1.attention.backends.mla.common.MLACommonImpl.__init__ = MetaxMLACommonImpl.__init__
vllm.v1.attention.backends.mla.common.MLACommonImpl._flash_attn_varlen_diff_headdims = MetaxMLACommonImpl._flash_attn_varlen_diff_headdims
vllm.v1.attention.backends.mla.common.MLACommonImpl.process_weights_after_loading = MetaxMLACommonImpl.process_weights_after_loading
vllm.v1.attention.backends.mla.common.MLACommonImpl._forward_prefill = MetaxMLACommonImpl._forward_prefill
vllm.v1.attention.backends.mla.common.MLACommonImpl.flash_attn_varlen_func = flash_attn_varlen_func

vllm.v1.attention.backends.mla.common.MLACommonMetadataBuilder.__init__ = MetaxMLACommonMetadataBuilder.__init__
vllm.v1.attention.backends.mla.common.MLACommonMetadataBuilder.build = MetaxMLACommonMetadataBuilder.build

register_patch("vllm.v1.attention.backends.mla.common", "get_flash_attn_version", get_flash_attn_version)

register_patch("vllm.v1.attention.backends.mla.common", "MLACommonImpl.__init__", MetaxMLACommonImpl.__init__)
register_patch("vllm.v1.attention.backends.mla.common", "MLACommonImpl._flash_attn_varlen_diff_headdims", MetaxMLACommonImpl._flash_attn_varlen_diff_headdims)
register_patch("vllm.v1.attention.backends.mla.common", "MLACommonImpl.process_weights_after_loading", MetaxMLACommonImpl.process_weights_after_loading)
register_patch("vllm.v1.attention.backends.mla.common", "MLACommonImpl._forward_prefill", MetaxMLACommonImpl._forward_prefill)
register_patch("vllm.v1.attention.backends.mla.common", "MLACommonImpl.flash_attn_varlen_func", flash_attn_varlen_func)

register_patch("vllm.v1.attention.backends.mla.common", "MLACommonMetadataBuilder.__init__",  MetaxMLACommonMetadataBuilder.__init__)
register_patch("vllm.v1.attention.backends.mla.common", "MLACommonMetadataBuilder.build", MetaxMLACommonMetadataBuilder.build)