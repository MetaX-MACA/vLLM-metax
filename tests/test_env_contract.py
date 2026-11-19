# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest

import vllm_metax.envs as mx_envs


@pytest.mark.parametrize(
    ("name", "default"),
    [
        ("MACA_VLLM_ENABLE_MCTLASS_FUSED_MOE", True),
        ("VLLM_METAX_OPTIMIZED_DP_ALL2ALL", True),
        ("VLLM_METAX_ENABLE_FA_SPLIT_FORWARD", True),
        ("VLLM_METAX_USE_FP8_WO_A", True),
        ("VLLM_METAX_USE_QUANT_MAIN_PROJ", True),
        ("VLLM_METAX_USE_SGL_FUSED_MOE_GROUPED_TOPK", False),
        ("VLLM_METAX_USE_FLASHINFER_SAMPLER", False),
        ("VLLM_METAX_DISABLE_ASYNC_OUTPUT", True),
        ("VLLM_METAX_ENABLE_MRV2_HWQ_BALANCING", False),
        ("USE_VLLM_TRITON_EXPERT", False),
    ],
)
def test_metax_environment_defaults(monkeypatch, name, default):
    monkeypatch.delenv(name, raising=False)
    assert name in mx_envs.environment_variables
    assert getattr(mx_envs, name) is default


@pytest.mark.parametrize(
    ("value", "expected"),
    [("0", False), ("1", True), ("true", False), ("false", False)],
)
def test_exact_one_environment_parser(monkeypatch, value, expected):
    monkeypatch.setenv("VLLM_METAX_ENABLE_MRV2_HWQ_BALANCING", value)
    assert mx_envs.VLLM_METAX_ENABLE_MRV2_HWQ_BALANCING is expected


@pytest.mark.parametrize("value", ["true", "false"])
def test_integer_boolean_environment_parser_rejects_non_integers(monkeypatch, value):
    monkeypatch.setenv("VLLM_METAX_ENABLE_FA_SPLIT_FORWARD", value)
    with pytest.raises(ValueError):
        _ = mx_envs.VLLM_METAX_ENABLE_FA_SPLIT_FORWARD


@pytest.mark.parametrize(("value", "expected"), [("2", True), ("-1", True)])
def test_integer_boolean_environment_parser_uses_int_truthiness(
    monkeypatch, value, expected
):
    monkeypatch.setenv("VLLM_METAX_ENABLE_FA_SPLIT_FORWARD", value)
    assert mx_envs.VLLM_METAX_ENABLE_FA_SPLIT_FORWARD is expected


def test_flashinfer_sampler_falls_back_to_upstream_switch(monkeypatch):
    monkeypatch.delenv("VLLM_METAX_USE_FLASHINFER_SAMPLER", raising=False)
    monkeypatch.setenv("VLLM_USE_FLASHINFER_SAMPLER", "1")
    assert mx_envs.VLLM_METAX_USE_FLASHINFER_SAMPLER is True


def test_is_set_tracks_explicit_environment(monkeypatch):
    monkeypatch.delenv("VLLM_METAX_ENABLE_FA_SPLIT_FORWARD", raising=False)
    assert mx_envs.is_set("VLLM_METAX_ENABLE_FA_SPLIT_FORWARD") is False
    monkeypatch.setenv("VLLM_METAX_ENABLE_FA_SPLIT_FORWARD", "0")
    assert mx_envs.is_set("VLLM_METAX_ENABLE_FA_SPLIT_FORWARD") is True


def test_maybe_override_vllm_env_updates_resolver(monkeypatch):
    from vllm import envs as upstream_envs

    name = "VLLM_METAX_ENABLE_MRV2_HWQ_BALANCING"
    monkeypatch.delenv(name, raising=False)
    monkeypatch.delitem(upstream_envs.environment_variables, name, raising=False)
    mx_envs.maybe_override_vllm_env(name, True, "contract test", forced=True)
    assert upstream_envs.environment_variables[name]() is True
