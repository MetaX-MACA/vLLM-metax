# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest
import torch

from vllm.config import CacheConfig, VllmConfig
from vllm.v1.attention.backends.utils import resolve_kv_cache_layout
from vllm.v1.core.kv_cache_config_builder import get_kv_cache_config_builder
from vllm.v1.core.kv_cache_planning import (
    DefaultKVCacheConfigBuilder,
    _project_kv_cache_groups_to_worker,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_metax.v1.core.kv_cache_config_builder import MacaKVCacheConfigBuilder


# Adapted from vLLM v0.33.0 tests/v1/core/test_kv_cache_config_builder.py.
@pytest.mark.parametrize("local_names", [("local_a",), ("local_a", "local_b")])
def test_plugin_builder_filters_remote_tensor_views(local_names):
    config = VllmConfig(cache_config=CacheConfig(block_size=16))
    local = FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=64, dtype=torch.bfloat16
    )
    remote = FullAttentionSpec(
        block_size=16, num_kv_heads=2, head_size=128, dtype=torch.bfloat16
    )
    resolve_kv_cache_layout(config, [["BLNHC"]], [local, remote])
    group_specs = [
        {"local_a": local, "local_b": local},
        {"remote_a": remote, "remote_b": remote},
    ]
    global_groups = [
        KVCacheGroupSpec(
            list(specs),
            UniformTypeKVCacheSpecs(block_size=16, kv_cache_specs=specs),
        )
        for specs in group_specs
    ]
    worker_groups = _project_kv_cache_groups_to_worker(
        global_groups, dict.fromkeys(local_names, local)
    )
    builder = get_kv_cache_config_builder(config)
    assert type(builder) is MacaKVCacheConfigBuilder
    actual = builder.get_kv_cache_config_from_groups(
        config, worker_groups, num_blocks=11
    )
    reference = DefaultKVCacheConfigBuilder().get_kv_cache_config_from_groups(
        config, worker_groups, num_blocks=11
    )

    assert actual.num_blocks == reference.num_blocks == 11
    assert actual.kv_cache_groups == reference.kv_cache_groups
    assert [group.layer_names for group in actual.kv_cache_groups] == [
        list(local_names),
        [],
    ]
    assert [
        name for tensor in actual.kv_cache_tensors for name in tensor.layers
    ] == list(local_names)
    assert any("remote_a" in tensor.layers for tensor in reference.kv_cache_tensors)
    assert actual.kv_cache_tensors == [
        tensor
        for tensor in reference.kv_cache_tensors
        if set(tensor.layers).intersection(local_names)
    ]
    for tensor in actual.kv_cache_tensors:
        assert tensor.block_stride == len(local_names) * local.page_size_bytes
