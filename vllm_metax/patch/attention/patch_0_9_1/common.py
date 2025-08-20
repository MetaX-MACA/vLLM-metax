# SPDX-License-Identifier: Apache-2.0

# TODO: remove this file

import vllm

from vllm_metax.patch.hook_registry import register_patch
from vllm import envs

########### MLACommonImpl nessessary #############
from vllm_metax.attention.backends.mla.common import MetaxMLACommonImpl, MetaxMLACommonMetadataBuilder
########## MLACommonImpl nessessary end ###########
    
from vllm.attention.backends.mla import common
vllm.attention.backends.mla.common.is_vllm_fa = False
vllm.attention.backends.mla.common.MLACommonMetadataBuilder.build = MetaxMLACommonMetadataBuilder.build
vllm.attention.backends.mla.common.MLACommonImpl.__init__ = MetaxMLACommonImpl.__init__
vllm.attention.backends.mla.common.MLACommonImpl.process_weights_after_loading = MetaxMLACommonImpl.process_weights_after_loading

register_patch("vllm.attention.backends.mla.common", "MLACommonMetadataBuilder.build", MetaxMLACommonMetadataBuilder.build)
register_patch("vllm.attention.backends.mla.common", "MLACommonImpl.__init__", MetaxMLACommonImpl.__init__)
register_patch("vllm.attention.backends.mla.common", "MLACommonImpl.process_weights_after_loading", MetaxMLACommonImpl.process_weights_after_loading)
