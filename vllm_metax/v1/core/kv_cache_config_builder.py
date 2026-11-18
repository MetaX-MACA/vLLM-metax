# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Keep PP-global allocation geometry while materializing local tensor views."""

from vllm.config import VllmConfig
from vllm.v1.core.kv_cache_planning import DefaultKVCacheConfigBuilder
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec


class MacaKVCacheConfigBuilder(DefaultKVCacheConfigBuilder):
    def get_kv_cache_config_from_groups(
        self,
        vllm_config: VllmConfig,
        kv_cache_groups: list[KVCacheGroupSpec],
        num_blocks: int,
    ) -> KVCacheConfig:
        config = super().get_kv_cache_config_from_groups(
            vllm_config, kv_cache_groups, num_blocks
        )
        # The target projector retains the full spec on a non-local group.
        # Keep global block sizing, offsets, IDs and all target layout handling;
        # only the subsequent worker's tensor-view materialization is filtered.
        local_layers = {name for group in kv_cache_groups for name in group.layer_names}
        config.kv_cache_tensors = [
            tensor
            for tensor in config.kv_cache_tensors
            if any(name in local_layers for name in tensor.layers)
        ]
        return config
