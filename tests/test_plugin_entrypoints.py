# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import importlib.metadata as metadata
import os
import sys

import pytest

import vllm_metax


def _entry_points(group: str):
    return {
        entry_point.name: entry_point
        for entry_point in metadata.entry_points().select(group=group)
        if entry_point.name.startswith("metax")
    }


def test_platform_entry_point_is_owned_by_plugin():
    entry_points = _entry_points("vllm.platform_plugins")
    assert entry_points["metax"].value == "vllm_metax:register"
    platform_path = entry_points["metax"].load()()
    assert platform_path == "vllm_metax.platform.MacaPlatform"


def test_general_plugin_entry_points_cover_plugin_registration_surfaces():
    entry_points = _entry_points("vllm.general_plugins")
    assert set(entry_points) == {
        "metax_attn_backends",
        "metax_enhanced_model",
        "metax_out_of_tree",
    }
    expected = {
        "metax_attn_backends": "vllm_metax:register_attn_backends",
        "metax_enhanced_model": "vllm_metax:register_model",
        "metax_out_of_tree": "vllm_metax:register_out_of_tree",
    }
    assert {name: ep.value for name, ep in entry_points.items()} == expected
    for entry_point in entry_points.values():
        plugin_hook = entry_point.load()
        assert callable(plugin_hook)
        plugin_hook()


def test_plugin_version_and_distribution_match():
    distribution = metadata.distribution("vllm_metax")
    assert distribution.version.startswith(vllm_metax.__version__)
    assert metadata.version("vllm_metax").startswith(vllm_metax.__version__)


@pytest.mark.skipif(
    os.environ.get("USE_PRECOMPILED_KERNEL", "0") != "0",
    reason="source-kernel validation only",
)
def test_source_mode_loads_plugin_extensions_without_mcoplib():
    assert not [name for name in sys.modules if name == "mcoplib" or name.startswith("mcoplib.")]

    import vllm_metax._C_stable_libtorch as core_extension
    import vllm_metax._moe_C_stable_libtorch as moe_extension

    assert core_extension.__file__.startswith(str(vllm_metax.__file__).rsplit("/", 1)[0])
    assert moe_extension.__file__.startswith(str(vllm_metax.__file__).rsplit("/", 1)[0])


def test_collect_env_console_entry_point_is_owned_by_plugin():
    entry_points = {
        entry_point.name: entry_point
        for entry_point in metadata.entry_points().select(group="console_scripts")
    }
    assert entry_points["vllm_collect_env"].value == "vllm_metax:collect_env"
