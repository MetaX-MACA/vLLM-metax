# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from types import SimpleNamespace

import pytest

from vllm_metax.utils.math_util import is_power_of_2
from vllm_metax.utils.mccl import find_mccl_library
from vllm_metax.utils.msprobe_debug import (
    build_precision_debugger,
    is_msprobe_dump_requested,
    resolve_worker_cls,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(1, True), (2, True), (4, True), (1024, True), (0, False), (3, False), (-4, False)],
)
def test_is_power_of_2(value, expected):
    assert is_power_of_2(value) is expected


def test_find_mccl_library_prefers_plugin_environment(monkeypatch, tmp_path):
    library = tmp_path / "libmccl-custom.so"
    monkeypatch.setenv("VLLM_MCCL_SO_PATH", str(library))
    assert find_mccl_library() == str(library)


def test_find_mccl_library_defaults_to_mccl_not_nccl(monkeypatch):
    monkeypatch.delenv("VLLM_MCCL_SO_PATH", raising=False)
    assert find_mccl_library() == "libmccl.so"


def test_worker_class_default_is_upstream_worker():
    assert resolve_worker_cls(None, enforce_eager=False) == (
        "vllm.v1.worker.gpu_worker.Worker"
    )


def test_msprobe_dump_selects_plugin_worker():
    config = {"dump_config_path": "/tmp/msprobe.json"}
    assert is_msprobe_dump_requested(config) is True
    assert resolve_worker_cls(config, enforce_eager=True) == (
        "vllm_metax.v1.worker.gpu_worker.MacaWorker"
    )


def test_msprobe_dump_requires_enforce_eager():
    config = {"dump_config_path": "/tmp/msprobe.json"}
    with pytest.raises(ValueError, match="requires --enforce-eager"):
        resolve_worker_cls(config, enforce_eager=False)


def test_msprobe_dump_config_absent_is_disabled():
    assert is_msprobe_dump_requested({"other": True}) is False
    assert build_precision_debugger({"other": True}) is None
