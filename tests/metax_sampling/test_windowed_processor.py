# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Adapter tests require an actual vLLM installation, not mocked interfaces."""

from types import SimpleNamespace
import json
import random

import pytest
import torch
from vllm.v1.sample.logits_processor.interface import BatchUpdate, MoveDirectionality

from vllm_metax.v1.sample.windowed_penalties import EXTRA_ARG
from vllm_metax.v1.sample.windowed_penalty_processor import (
    WindowedPenaltyLogitsProcessor,
)


def params(policy=None, **kwargs):
    return SimpleNamespace(
        extra_args={} if policy is None else {EXTRA_ARG: policy},
        repetition_penalty=kwargs.get("repetition_penalty", 1.0),
        frequency_penalty=kwargs.get("frequency_penalty", 0.0),
        presence_penalty=kwargs.get("presence_penalty", 0.0),
    )


def update(*, added=(), removed=(), moved=(), size=2):
    return BatchUpdate(batch_size=size, removed=removed, added=added, moved=moved)


def test_live_history_and_slot_reuse():
    processor = WindowedPenaltyLogitsProcessor(None, torch.device("cpu"), False)
    policy = params({"window": 2, "frequency": 1})
    history = [1]
    processor.update_state(update(added=[(0, policy, [7, 7], history)]))
    history.extend([2, 2])
    processor.update_state(None)
    assert processor.apply(torch.ones(2, 8))[0].tolist() == [1, 1, -1, 1, 1, 1, 1, 1]
    processor.update_state(update(removed=[0], added=[(0, policy, None, [4])]))
    assert processor.apply(torch.ones(2, 8))[0, 4] == 0
    processor.update_state(update(added=[(0, params(), None, [])]))
    assert not processor.requests
    assert processor.batch.num_updates == 0


@pytest.mark.parametrize("direction", list(MoveDirectionality))
@pytest.mark.parametrize(
    "active_source,active_target", [(True, True), (False, True), (True, False)]
)
def test_moves_and_swaps(direction, active_source, active_target):
    processor = WindowedPenaltyLogitsProcessor(None, torch.device("cpu"), False)
    enabled = params({"window": 2, "frequency": 1})
    processor.update_state(
        update(
            added=[
                (0, enabled if active_source else params(), None, [2]),
                (1, enabled if active_target else params(), None, [3]),
            ]
        )
    )
    processor.update_state(update(moved=[(0, 1, direction)]))
    result = processor.apply(torch.ones(2, 8))
    assert result[1, 2] == (0 if active_source else 1)
    assert result[0, 3] == (
        0 if active_target and direction == MoveDirectionality.SWAP else 1
    )
    assert result[0, 2] == 1
    assert result[1, 3] == 1


def test_self_move_and_remove_everything():
    processor = WindowedPenaltyLogitsProcessor(None, torch.device("cpu"), False)
    processor.update_state(
        update(added=[(0, params({"window": 2, "frequency": 1}), None, [1])])
    )
    processor.update_state(update(moved=[(0, 0, MoveDirectionality.SWAP)]))
    assert processor.apply(torch.ones(2, 8))[0, 1] == 0
    processor.update_state(update(removed=[0], size=0))
    assert not processor.requests
    assert processor.apply(torch.empty(0, 8)).shape == (0, 8)
    assert not processor.is_argmax_invariant()


def test_history_is_read_after_async_placeholder_replacement():
    processor = WindowedPenaltyLogitsProcessor(None, torch.device("cpu"), False)
    history = [1, -1]
    processor.update_state(
        update(added=[(0, params({"window": 2, "frequency": 1}), None, history)])
    )
    history[-1] = 2
    assert processor.apply(torch.ones(2, 8))[0].tolist() == [1, 0, 0, 1, 1, 1, 1, 1]
    history.append(-1)
    processor.update_state(None)
    history[-1] = 2
    assert processor.apply(torch.ones(2, 8))[0].tolist() == [1, 1, -1, 1, 1, 1, 1, 1]
    history[:] = [3]
    assert processor.apply(torch.ones(2, 8))[0].tolist() == [1, 1, 1, 0, 1, 1, 1, 1]


@pytest.mark.parametrize(
    "field,value",
    [
        ("repetition_penalty", 1.2),
        ("frequency_penalty", 0.5),
        ("presence_penalty", 0.5),
    ],
)
def test_reject_double_penalties(field, value):
    with pytest.raises(ValueError, match="neutral"):
        WindowedPenaltyLogitsProcessor.validate_params(
            params({"window": 2, "repetition": 1.2}, **{field: value})
        )
    WindowedPenaltyLogitsProcessor.validate_params(params(None, **{field: value}))


def test_actual_sampling_params_and_loader():
    from vllm import SamplingParams
    from vllm.v1.sample.logits_processor import _load_logitsprocs_by_fqcns

    classes = _load_logitsprocs_by_fqcns(
        [
            "vllm_metax.v1.sample.windowed_penalty_processor:WindowedPenaltyLogitsProcessor"
        ]
    )
    assert classes == [WindowedPenaltyLogitsProcessor]
    sampling = SamplingParams(
        extra_args={EXTRA_ARG: {"window": 4, "repetition": 1.2}}, temperature=0
    )
    WindowedPenaltyLogitsProcessor.validate_params(sampling)


def test_v2_loader_rejects_v1_adapter_explicitly():
    from vllm.v1.worker.gpu.sample.logits_processor.loader import (
        _load_v2_logitsprocs_by_fqcns,
    )

    with pytest.raises(ValueError, match="not a subclass"):
        _load_v2_logitsprocs_by_fqcns(
            [
                "vllm_metax.v1.sample.windowed_penalty_processor:WindowedPenaltyLogitsProcessor"
            ]
        )


def test_openai_request_roundtrip_and_validation():
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.exceptions import VLLMValidationError
    from vllm.v1.sample.logits_processor import validate_logits_processors_parameters

    fqcn = (
        "vllm_metax.v1.sample.windowed_penalty_processor:WindowedPenaltyLogitsProcessor"
    )
    policy = {
        "window": 4,
        "frequency": 0.5,
        "token_ranges": [[2, 8]],
        "exempt_token_ids": [3],
    }
    request = ChatCompletionRequest.model_validate(
        {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hello"}],
            "vllm_xargs": {EXTRA_ARG: json.dumps(policy)},
            "temperature": 0,
        }
    )
    sampling = request.to_sampling_params(max_tokens=8, default_sampling_params={})
    validate_logits_processors_parameters([fqcn], sampling)
    processor = WindowedPenaltyLogitsProcessor(None, torch.device("cpu"), False)
    processor.update_state(update(added=[(0, sampling, [1, 7], [1, 2, 2, 3])]))
    assert processor.apply(torch.ones(2, 8))[0].tolist() == [1, 1, 0, 1, 1, 1, 1, 1]
    sampling.extra_args[EXTRA_ARG] = '{"window":0}'
    with pytest.raises(VLLMValidationError):
        validate_logits_processors_parameters([fqcn], sampling)


@pytest.fixture
def production_sampler_case():
    from vllm import SamplingParams
    from vllm.config import DeviceConfig, VllmConfig
    from vllm.v1.sample.logits_processor import build_logitsprocs
    from vllm.v1.sample.metadata import SamplingMetadata
    from vllm.v1.sample.sampler import Sampler

    config = VllmConfig(device_config=DeviceConfig(device="cpu"))
    processors = build_logitsprocs(
        config,
        torch.device("cpu"),
        False,
        False,
        [
            "vllm_metax.v1.sample.windowed_penalty_processor:WindowedPenaltyLogitsProcessor"
        ],
    )
    policy = SamplingParams(
        temperature=0, extra_args={EXTRA_ARG: {"window": 4, "repetition": 2}}
    )
    plain = SamplingParams(temperature=0)
    history = [[1], [1]]
    delta = update(added=[(0, policy, [], history[0]), (1, plain, [], history[1])])
    for processor in processors.all:
        processor.update_state(delta)
    metadata = SamplingMetadata(
        temperature=None,
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.zeros(2),
        presence_penalties=torch.zeros(2),
        repetition_penalties=torch.ones(2),
        output_token_ids=history,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=processors,
    )
    logits = torch.tensor([[0.0, 8.0, 6.0], [0.0, 8.0, 6.0]])
    return Sampler(), logits, metadata


def test_production_logits_chain_only_changes_opted_in_row(production_sampler_case):
    sampler, logits, metadata = production_sampler_case
    actual = sampler.apply_logits_processors(logits, metadata, False)
    torch.testing.assert_close(actual, torch.tensor([[0.0, 4.0, 6.0], [0.0, 8.0, 6.0]]))
    assert actual.argmax(dim=-1).tolist() == [2, 1]


def test_production_sampler_changes_greedy_choice_only_for_opted_in_row(
    production_sampler_case,
):
    from vllm.platforms import current_platform

    if current_platform.is_cpu() and not hasattr(torch.ops._C, "greedy_argmax"):
        pytest.skip(
            "full CPU sampler requires vLLM native _C.greedy_argmax; empty builds only validate the logits chain"
        )
    sampler, logits, metadata = production_sampler_case
    result = sampler(logits, metadata)
    assert result.sampled_token_ids.flatten().tolist() == [2, 1]


def test_real_sampler_observes_async_history_and_preserves_raw_logprobs(
    production_sampler_case,
):
    from vllm.platforms import current_platform

    if current_platform.is_cpu() and not hasattr(torch.ops._C, "greedy_argmax"):
        pytest.skip("requires the real upstream CPU greedy_argmax operator")
    sampler, logits, metadata = production_sampler_case
    metadata.max_num_logprobs = -1
    raw = logits.log_softmax(dim=-1)
    for expected_tokens in ([2, 1], [1, 1], [1, 1], [1, 1], [1, 1], [2, 1]):
        output = sampler(logits.clone(), metadata)
        assert output.sampled_token_ids.flatten().tolist() == expected_tokens
        torch.testing.assert_close(output.logprobs_tensors.logprobs, raw)
        for history in metadata.output_token_ids:
            history.append(-1)
        for processor in metadata.logitsprocs.all:
            processor.update_state(None)
        for history, token in zip(
            metadata.output_token_ids, expected_tokens, strict=True
        ):
            history[-1] = token


def test_randomized_scheduler_lifecycle_against_independent_reference():
    rng = random.Random(42)
    processor = WindowedPenaltyLogitsProcessor(None, torch.device("cpu"), False)
    states = {}
    policy = params({"window": 3, "frequency": 0.5})
    for _ in range(100):
        index = rng.randrange(4)
        operation = rng.randrange(5)
        if operation == 0:
            states.pop(index, None)
            processor.update_state(update(removed=[index], size=4))
        elif operation == 1:
            history = [rng.randrange(8) for _ in range(rng.randrange(5))]
            states[index] = history
            processor.update_state(
                update(removed=[index], added=[(index, policy, [], history)], size=4)
            )
        elif operation == 2:
            states.pop(index, None)
            processor.update_state(update(added=[(index, params(), [], [])], size=4))
        elif operation == 3:
            target = (index + 1) % 4
            before = dict(states)
            states.pop(index, None)
            states.pop(target, None)
            direction = rng.choice(list(MoveDirectionality))
            if index in before:
                states[target] = before[index]
            if target in before and direction == MoveDirectionality.SWAP:
                states[index] = before[target]
            processor.update_state(update(moved=[(index, target, direction)], size=4))
        else:
            for history in states.values():
                if history and rng.randrange(3) == 0:
                    history.pop()
                else:
                    history.append(rng.randrange(8))
            processor.update_state(None)
        expected = torch.ones(4, 8)
        for row, history in states.items():
            for token in history[-3:]:
                expected[row, token] -= 0.5
        torch.testing.assert_close(processor.apply(torch.ones(4, 8)), expected)
