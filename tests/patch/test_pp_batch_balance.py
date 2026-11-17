# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import ast
import inspect
import textwrap
from types import SimpleNamespace

from vllm_metax.patch.enhancement.performance import pp_batch_balance


def test_transform_matches_target_v33_schedule_shape():
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


def test_balanced_schedule_compiles_for_target_schedule():
    schedule = pp_batch_balance._schedule_with_balance(
        pp_batch_balance._original_schedule
    )

    assert schedule.__name__ == "schedule"
    assert callable(schedule)


def test_state_defers_only_excess_decodes_and_never_starves():
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
    assert state.target == 2

    first, second, third = requests.values()
    assert not state.defer(first, 1, 4, set())
    state.record(first, 1, 4)
    assert not state.defer(second, 1, 4, {"0"})
    state.record(second, 1, 4)
    assert state.defer(third, 1, 4, {"0", "1"})

    state.deferred["2"] = state.pp_size
    assert not state.defer(third, 1, 4, {"0", "1"})


def test_balancing_is_scoped_to_synchronous_v2_pp_fcfs():
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
    assert pp_batch_balance._supports_balancing(config)

    config.scheduler_config.async_scheduling = True
    assert not pp_batch_balance._supports_balancing(config)


# These cases exercise the installed Scheduler, KV manager and output lifecycle.
# No model or worker is constructed.
import json

import pytest
import torch

from vllm.config import CacheConfig, ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_utils import init_none_hash, get_request_block_hasher
from vllm.utils.hashing import sha256
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import register_all_kvcache_specs
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output import StructuredOutputManager


@pytest.fixture
def scheduler_factory(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    (tmp_path / "config.json").write_text(json.dumps({
        "model_type": "opt", "architectures": ["OPTForCausalLM"],
        "hidden_size": 32, "ffn_dim": 64, "num_hidden_layers": 2,
        "num_attention_heads": 2, "vocab_size": 32,
        "max_position_embeddings": 128, "torch_dtype": "float16"
    }))
    def make(*, pp=2, capacity=8, active=None, budget=64, blocks=128, prefix=False):
        config = VllmConfig(
            model_config=ModelConfig(model=str(tmp_path), dtype="float16", max_model_len=128, skip_tokenizer_init=True),
            scheduler_config=SchedulerConfig(max_num_seqs=capacity, max_num_active_seqs=active, max_num_batched_tokens=budget, max_model_len=128, enable_chunked_prefill=True, async_scheduling=False, policy="fcfs", watermark=0.0, is_encoder_decoder=False),
            cache_config=CacheConfig(block_size=2, enable_prefix_caching=prefix),
            parallel_config=ParallelConfig(pipeline_parallel_size=pp, data_parallel_size=1),
        )
        config.cache_config.num_gpu_blocks = blocks
        register_all_kvcache_specs(config)
        kv = KVCacheConfig(num_blocks=blocks, kv_cache_tensors=[], kv_cache_groups=[KVCacheGroupSpec(["layer"], FullAttentionSpec(block_size=2, num_kv_heads=1, head_size=16, dtype=torch.float16))])
        return Scheduler(vllm_config=config, kv_cache_config=kv, block_size=2, log_stats=False, structured_output_manager=StructuredOutputManager(config))
    return make


def _request(rid, max_tokens=5):
    init_none_hash(sha256)
    params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
    params.update_from_generation_config({}, 31)
    return Request(request_id=rid, prompt_token_ids=[1, 2, 3, 4], sampling_params=params, pooling_params=None, block_hasher=get_request_block_hasher(2, sha256))


def _deliver(scheduler, output):
    ids = list(output.num_scheduled_tokens)
    scheduler.update_from_output(output, ModelRunnerOutput(req_ids=ids, req_id_to_index={rid: i for i, rid in enumerate(ids)}, sampled_token_ids=[[9] for _ in ids], logprobs=None, prompt_logprobs_dict={}, pooler_output=[]))


def test_real_scheduler_counts_inflight_requests_and_balances_two_batches(scheduler_factory):
    scheduler = scheduler_factory()
    requests = [_request(str(i)) for i in range(4)]
    for request in requests:
        scheduler.add_request(request)
    prefill = scheduler.schedule()
    assert prefill.num_scheduled_tokens == {str(i): 4 for i in range(4)}
    _deliver(scheduler, prefill)
    first = scheduler.schedule()
    second = scheduler.schedule()
    assert scheduler._metax_pp_balance.target == 2
    assert len(first.num_scheduled_tokens) == len(second.num_scheduled_tokens) == 2
    assert set(first.num_scheduled_tokens).isdisjoint(second.num_scheduled_tokens)
    assert set(first.num_scheduled_tokens) | set(second.num_scheduled_tokens) == set(scheduler.requests)
    assert all(value == 1 for output in [first, second] for value in output.num_scheduled_tokens.values())
    _deliver(scheduler, first)
    _deliver(scheduler, second)
    assert all(request.num_in_flight_tokens == 0 for request in requests)


def test_real_scheduler_drains_without_starvation_and_preserves_arrival_prefill(scheduler_factory):
    scheduler = scheduler_factory()
    requests = [_request(str(i), max_tokens=4) for i in range(4)]
    for request in requests:
        scheduler.add_request(request)
    _deliver(scheduler, scheduler.schedule())
    maximum_gap = {request.request_id: 0 for request in requests}
    gaps = maximum_gap.copy()
    for step in range(20):
        if step == 1:
            scheduler.add_request(_request("arrival", max_tokens=2))
        if not scheduler.has_unfinished_requests():
            break
        ready = {rid for rid, req in scheduler.requests.items() if req.num_computed_tokens >= req.num_prompt_tokens}
        output = scheduler.schedule()
        assert sum(output.num_scheduled_tokens.values()) <= 64
        if step == 1:
            assert output.num_scheduled_tokens["arrival"] == 4
        for rid in ready & gaps.keys():
            gaps[rid] = 0 if rid in output.num_scheduled_tokens else gaps[rid] + 1
            maximum_gap[rid] = max(maximum_gap[rid], gaps[rid])
        _deliver(scheduler, output)
    assert not scheduler.has_unfinished_requests()
    assert all(request.num_output_tokens == 4 for request in requests)
    assert max(maximum_gap.values()) <= 2
    # A later call prunes all completed request IDs from the deferral state.
    scheduler.schedule()
    assert not scheduler._metax_pp_balance.deferred


def test_real_scheduler_keeps_v33_active_admission_budget(scheduler_factory):
    scheduler = scheduler_factory(active=2)
    for i in range(4):
        scheduler.add_request(_request(str(i)))
    output = scheduler.schedule(throttle_prefills=False)
    assert len(output.num_scheduled_tokens) == 2
    assert len(scheduler.running) == 2
    assert len(scheduler.waiting) == 2


def test_single_stage_uses_original_scheduler(scheduler_factory):
    scheduler = scheduler_factory(pp=1)
    for i in range(4):
        scheduler.add_request(_request(str(i)))
    prefill = scheduler.schedule()
    _deliver(scheduler, prefill)
    output = scheduler.schedule()
    assert len(output.num_scheduled_tokens) == 4
    assert not hasattr(scheduler, "_metax_pp_balance")



def test_waiting_cached_decodes_obey_the_same_balance_target(scheduler_factory):
    scheduler = scheduler_factory(prefix=True)
    scheduler.add_request(_request("warm"))
    _deliver(scheduler, scheduler.schedule())
    for i in range(3):
        request = _request(f"cached-{i}")
        request.append_output_token_ids([9])
        scheduler.add_request(request)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {"warm": 1, "cached-0": 1}
    assert len(scheduler.waiting) == 2
    assert scheduler._metax_pp_balance.target == 2
    _deliver(scheduler, output)


def test_real_preemption_keeps_the_decode_cap_and_request_accounting(scheduler_factory):
    # Nine blocks include the manager's null block, leaving exactly eight
    # blocks for four prompts. Decode then forces actual KV preemption.
    scheduler = scheduler_factory(blocks=9)
    requests = [_request(str(i)) for i in range(4)]
    for request in requests:
        scheduler.add_request(request)
    prefill = scheduler.schedule()
    assert len(prefill.num_scheduled_tokens) == 4
    _deliver(scheduler, prefill)
    output = scheduler.schedule()
    assert any(request.num_preemptions for request in requests)
    assert len(output.num_scheduled_tokens) == 2
    assert all(value == 1 for value in output.num_scheduled_tokens.values())
    assert set(scheduler.requests) == {str(i) for i in range(4)}
    assert len(scheduler.running) + len(scheduler.waiting) == 4
    _deliver(scheduler, output)


@pytest.mark.parametrize("section,attribute,value", [
    ("parallel_config", "pipeline_parallel_size", 1),
    ("parallel_config", "data_parallel_size", 2),
    ("scheduler_config", "policy", "priority"),
    (None, "speculative_config", object()),
    (None, "use_v2_model_runner", False),
])
def test_other_configurations_are_not_balanced(section, attribute, value):
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(pipeline_parallel_size=2, data_parallel_size=1),
        scheduler_config=SimpleNamespace(async_scheduling=False, policy="fcfs"),
        speculative_config=None,
        use_v2_model_runner=True,
    )
    setattr(config if section is None else getattr(config, section), attribute, value)
    assert not pp_batch_balance._supports_balancing(config)
