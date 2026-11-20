# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
from importlib import metadata

import vllm_metax.collect_env as collect_env


def test_plugin_version_is_reported():
    assert collect_env.get_vllm_metax_version() == metadata.version("vllm-metax")


def test_maca_sdk_version_reads_maca_version_file(monkeypatch):
    observed = []

    def run_lambda(command):
        observed.append(command)
        return 0, "Version:3.10.1.11", ""

    monkeypatch.setenv("MACA_PATH", "/opt/maca")
    assert collect_env.get_maca_sdk_version(run_lambda) == "3.10.1.11"
    assert observed == [["cat", "/opt/maca/Version.txt"]]


def test_gpu_topology_uses_mx_smi_on_linux(monkeypatch):
    observed = []

    def run_lambda(command):
        observed.append(command)
        return 0, "MACA topology", ""

    monkeypatch.setattr(collect_env, "get_platform", lambda: "linux")
    assert collect_env.get_gpu_topo(run_lambda) == "MACA topology"
    assert observed == ["mx-smi topo -m"]


def test_environment_report_captures_maca_and_filters_secrets(monkeypatch):
    monkeypatch.setenv("MACA_PATH", "/opt/maca")
    monkeypatch.setenv("VLLM_METAX_ENABLE_FA_SPLIT_FORWARD", "0")
    monkeypatch.setenv("MACA_API_TOKEN", "do-not-print")
    monkeypatch.setenv("UNRELATED_ENV", "ignored")
    report = collect_env.get_env_vars()
    assert "MACA_PATH=/opt/maca" in report
    assert "VLLM_METAX_ENABLE_FA_SPLIT_FORWARD=0" in report
    assert "MACA_API_TOKEN" not in report
    assert "UNRELATED_ENV" not in report
