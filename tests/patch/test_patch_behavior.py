# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import ast
import inspect
import sys
import textwrap
from types import SimpleNamespace

import pytest

from vllm_metax.patch.enhancement.performance import pp_batch_balance
from vllm_metax.patch.models.kimi_k3 import disable_gemm_rs_ar


def test_pp_balance_transforms_target_schedule():
    lines, _ = inspect.getsourcelines(pp_batch_balance._original_schedule)
    tree = ast.parse(textwrap.dedent("".join(lines)))
    transformed = pp_batch_balance._transform_schedule(tree)
    calls = [
        node
        for node in ast.walk(transformed)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "_metax_pp_balance"
    ]
    assert [node.func.attr for node in calls].count("defer") == 2
    assert [node.func.attr for node in calls].count("record") == 2


def test_pp_balance_compiles_for_target_schedule():
    schedule = pp_batch_balance._schedule_with_balance(
        pp_batch_balance._original_schedule
    )
    assert schedule.__name__ == "schedule"
    assert callable(schedule)


def test_pp_balance_defers_excess_decodes_without_starvation():
    state = pp_batch_balance._PPBatchBalance(pp_size=2)
    requests = {
        idx: SimpleNamespace(request_id=str(idx), num_prompt_tokens=4)
        for idx in range(3)
    }
    scheduler = SimpleNamespace(
        max_num_running_reqs=8,
        get_num_unfinished_requests=lambda: 4,
        requests=requests,
    )
    state.begin(scheduler)
    first, second, third = requests.values()
    assert not state.defer(first, 1, 4, set())
    state.record(first, 1, 4)
    assert not state.defer(second, 1, 4, {"0"})
    state.record(second, 1, 4)
    assert state.defer(third, 1, 4, {"0", "1"})


@pytest.mark.parametrize(
    ("section", "attribute", "value"),
    [
        ("parallel_config", "pipeline_parallel_size", 1),
        ("parallel_config", "data_parallel_size", 2),
        ("scheduler_config", "async_scheduling", True),
        ("scheduler_config", "policy", "priority"),
        (None, "speculative_config", object()),
        (None, "use_v2_model_runner", False),
    ],
)
def test_pp_balance_scope_rejects_incompatible_configuration(
    section, attribute, value
):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=2,
            data_parallel_size=1,
        ),
        scheduler_config=SimpleNamespace(
            async_scheduling=False,
            policy="fcfs",
        ),
        speculative_config=None,
        use_v2_model_runner=True,
    )
    setattr(config if section is None else getattr(config, section), attribute, value)
    assert not pp_batch_balance._supports_balancing(config)


@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize(
    ("ar_enabled", "rs_enabled"),
    [("0", "0"), ("1", "0"), ("0", "1"), ("1", "1")],
)
def test_kimi_gemm_rs_ar_is_disabled_on_maca(
    monkeypatch, sequence_parallel, ar_enabled, rs_enabled
):
    from vllm.models.kimi_k3.nvidia import model

    monkeypatch.setenv("VLLM_KIMI_K3_GEMM_AR", ar_enabled)
    monkeypatch.setenv("VLLM_ENABLE_GEMM_RS", rs_enabled)
    monkeypatch.setitem(
        sys.modules,
        "vllm.model_executor.kernels.linear.cute_dsl.gemm_rs_ar",
        None,
    )
    assert model.maybe_init_gemm_rs_ar is disable_gemm_rs_ar._maybe_init_gemm_rs_ar
    assert (
        model.maybe_init_gemm_rs_ar(None, use_sequence_parallel=sequence_parallel)
        is False
    )


def test_kimi_patch_preserves_target_signature():
    from vllm.models.kimi_k3.nvidia import model

    assert list(inspect.signature(model.maybe_init_gemm_rs_ar).parameters) == [
        "vllm_config",
        "use_sequence_parallel",
    ]
