# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Run independently: pytest --confcutdir=tests/metax_sampling tests/metax_sampling."""

import random
import json

import pytest
import torch

from vllm_metax.v1.sample.windowed_penalties import (
    SparsePenaltyBatch,
    WindowedPenaltyConfig,
    WindowedPenaltyState,
)


def dense_reference(logits, requests):
    result = logits.clone()
    for row, state in requests.items():
        config = state.config
        if row < 0 or row >= logits.shape[0]:
            continue
        history = state.output_token_ids[-config.window :]
        for token in set(history):
            if not 0 <= token < logits.shape[1] or not config.includes(token):
                continue
            original = result[row, token].float()
            penalized = (
                original / config.repetition
                if original > 0
                else original * config.repetition
            )
            result[row, token] = (
                penalized - config.frequency * history.count(token) - config.presence
            )
    return result


@pytest.mark.parametrize(
    "options",
    [
        None,
        [],
        {},
        {"window": 0},
        {"window": -1},
        {"window": True},
        {"window": 1.5},
        {"window": 65537},
        {"window": 2, "typo": 1},
        {"window": 2, "repetition": 0},
        {"window": 2, "repetition": float("nan")},
        {"window": 2, "repetition": float("inf")},
        {"window": 2, "repetition": True},
        {"window": 2, "repetition": 10**500},
        {"window": 2, "frequency": 3},
        {"window": 2, "presence": -3},
        {"window": 2, "presence": "1"},
        {"window": 2, "token_ranges": [[2, 2]]},
        {"window": 2, "token_ranges": [[0, 3], [2, 5]]},
        {"window": 2, "token_ranges": [[True, 5]]},
        {"window": 2, "token_ranges": [[1, 4, 5]]},
        {"window": 2, "token_ranges": "0:5"},
        {"window": 2, "exempt_token_ids": [-1]},
        {"window": 2, "exempt_token_ids": [True]},
        {"window": 2, "exempt_token_ids": "1"},
        {"window": 2, 1: "bad", None: "bad"},
        "{}",
        "[]",
        "null",
        "{",
        "1",
        "x" * 131073,
        "[" * 2000 + "]" * 2000,
    ],
)
def test_invalid_config(options):
    with pytest.raises(ValueError):
        WindowedPenaltyConfig.parse(options)


def test_config_is_detached_from_caller():
    options = {
        "window": 4,
        "repetition": 1.2,
        "token_ranges": [[4, 8], [0, 3]],
        "exempt_token_ids": [1],
    }
    config = WindowedPenaltyConfig.parse(options)
    options["token_ranges"][0][0] = 99
    options["exempt_token_ids"].append(2)
    assert config.token_ranges == ((0, 3), (4, 8))
    assert config.includes(2)
    assert not config.includes(1)
    assert not config.includes(3)
    assert not config.includes(8)


def test_json_string_matches_offline_policy():
    options = {
        "window": 4,
        "repetition": 1.2,
        "frequency": -0.5,
        "presence": 0.2,
        "token_ranges": [[3, 8]],
        "exempt_token_ids": [4],
    }
    assert WindowedPenaltyConfig.parse(
        json.dumps(options)
    ) == WindowedPenaltyConfig.parse(options)


@pytest.mark.parametrize(
    "options",
    [
        {"window": 1, "token_ranges": [[i, i + 1] for i in range(257)]},
        {"window": 1, "exempt_token_ids": list(range(4097))},
    ],
)
def test_bounded_config_metadata(options):
    with pytest.raises(ValueError):
        WindowedPenaltyConfig.parse(options)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("window", [1, 4, 32])
@pytest.mark.parametrize("scoped", [False, True])
@pytest.mark.parametrize("repetition", [0.8, 1.0, 1.2])
def test_cpu_matches_reference(dtype, window, scoped, repetition):
    rng = random.Random(12)
    torch.manual_seed(12)
    options = {
        "window": window,
        "repetition": repetition,
        "frequency": 0.25,
        "presence": -0.5,
    }
    if scoped:
        options.update(token_ranges=[[0, 7], [10, 16]], exempt_token_ids=[3, 11])
    config = WindowedPenaltyConfig.parse(options)
    requests = {
        row: WindowedPenaltyState(config, [rng.randrange(17) for _ in range(40)])
        for row in (0, 2)
    }
    logits = torch.randn(4, 17, dtype=dtype)
    logits[0, 0] = -float("inf")
    batch = SparsePenaltyBatch(torch.device("cpu"))
    batch.update(requests)
    expected = dense_reference(logits, requests)
    assert batch.apply(logits) is logits
    torch.testing.assert_close(logits, expected)
    assert batch.num_updates <= len(requests) * window


def test_repeated_ids_are_penalized_once():
    config = WindowedPenaltyConfig.parse(
        {"window": 4, "repetition": 2, "frequency": 0.5, "presence": 1}
    )
    batch = SparsePenaltyBatch(torch.device("cpu"))
    batch.update({0: WindowedPenaltyState(config, [1, 1, 1, 1])})
    logits = torch.tensor([[9.0, 8.0, 7.0]])
    batch.apply(logits)
    assert logits.tolist() == [[9.0, 1.0, 7.0]]
    assert batch.num_updates == 1


def test_window_evicts_and_rebuilds_after_rollback():
    history = [1, 2, 2]
    config = WindowedPenaltyConfig.parse({"window": 2, "frequency": 1})
    batch = SparsePenaltyBatch(torch.device("cpu"))
    state = {0: WindowedPenaltyState(config, history)}
    for replacement in ([1, 2, 2], [1, 2, 2, 3], [4], [5]):
        history[:] = replacement
        batch.update(state)
        logits = torch.ones(1, 8)
        torch.testing.assert_close(
            batch.apply(logits.clone()), dense_reference(logits, state)
        )


def test_ineligible_tokens_still_consume_window_and_exemptions_win():
    config = WindowedPenaltyConfig.parse(
        {"window": 3, "frequency": 1, "token_ranges": [[1, 4]], "exempt_token_ids": [2]}
    )
    batch = SparsePenaltyBatch(torch.device("cpu"))
    batch.update({0: WindowedPenaltyState(config, [1, 1, 1, 2, 3, 7])})
    assert batch.apply(torch.ones(1, 8)).tolist() == [[1, 1, 1, 0, 1, 1, 1, 1]]
    assert batch.num_updates == 1


def test_mixed_request_policies_keep_independent_windows():
    batch = SparsePenaltyBatch(torch.device("cpu"))
    batch.update(
        {
            0: WindowedPenaltyState(
                WindowedPenaltyConfig.parse({"window": 1, "repetition": 2}), [1, 2]
            ),
            1: WindowedPenaltyState(
                WindowedPenaltyConfig.parse(
                    {"window": 3, "frequency": 0.5, "presence": 1}
                ),
                [2, 1, 1],
            ),
        }
    )
    logits = torch.tensor([[0.0, 8.0, -8.0], [0.0, 8.0, -8.0]])
    assert batch.apply(logits).tolist() == [[0.0, 8.0, -16.0], [0.0, 6.0, -9.5]]


def test_noop_and_clear_preserve_identity():
    batch = SparsePenaltyBatch(torch.device("cpu"))
    logits = torch.ones(2, 8)
    assert batch.apply(logits) is logits
    config = WindowedPenaltyConfig.parse({"window": 4, "repetition": 2})
    batch.update({0: WindowedPenaltyState(config, [1])})
    batch.update({})
    assert batch.num_updates == 0
    torch.testing.assert_close(batch.apply(logits), torch.ones_like(logits))


def test_neutral_policy_does_not_allocate_updates():
    batch = SparsePenaltyBatch(torch.device("cpu"))
    config = WindowedPenaltyConfig.parse({"window": 4})
    batch.update({0: WindowedPenaltyState(config, [1, 2, 3])})
    assert batch.num_updates == 0


@pytest.mark.parametrize("shape,dtype", [((8,), torch.float32), ((2, 8), torch.int32)])
def test_invalid_logits(shape, dtype):
    batch = SparsePenaltyBatch(torch.device("cpu"))
    batch.update(
        {
            0: WindowedPenaltyState(
                WindowedPenaltyConfig.parse({"window": 1, "frequency": 1}), [1]
            )
        }
    )
    with pytest.raises(ValueError, match="2D"):
        batch.apply(torch.ones(shape, dtype=dtype))


def test_gpu_scoped_policies_and_rollback():
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    policies = [
        {
            "window": 4,
            "repetition": 0.8,
            "frequency": -0.5,
            "presence": 0.2,
            "token_ranges": [[2, 8]],
            "exempt_token_ids": [3],
        },
        {"window": 2, "repetition": 2, "frequency": 0.5, "presence": -0.2},
    ]
    requests = {
        i: WindowedPenaltyState(WindowedPenaltyConfig.parse(p), [1, 2, 3, 4, 4])
        for i, p in enumerate(policies)
    }
    batch = SparsePenaltyBatch(torch.device("cuda"), True)
    for replacement in ([1, 2, 3, 4, 4], [5, 3], [6, 3], []):
        for state in requests.values():
            state.output_token_ids[:] = replacement
        batch.update(requests)
        logits = torch.tensor(
            [[1.0, -1.0, 2.0, -float("inf"), -3.0, 0.0, 4.0, 8.0]] * 2, device="cuda"
        )
        expected = dense_reference(logits.cpu(), requests)
        torch.testing.assert_close(batch.apply(logits).cpu(), expected)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_strided_logits_and_out_of_bounds_ids(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    config = WindowedPenaltyConfig.parse({"window": 8, "repetition": 1.3})
    requests = {
        0: WindowedPenaltyState(config, [-1, 2, 30]),
        99: WindowedPenaltyState(config, [2]),
    }
    batch = SparsePenaltyBatch(torch.device(device))
    batch.update(requests)
    backing = torch.arange(64, dtype=torch.float32, device=device).reshape(2, 32)
    logits = backing[:, ::2]
    untouched = backing[:, 1::2].clone()
    expected = dense_reference(logits, requests)
    batch.apply(logits)
    torch.testing.assert_close(logits, expected)
    torch.testing.assert_close(backing[:, 1::2], untouched)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "batch_size,vocab,window", [(1, 17, 1), (4, 1019, 32), (32, 151936, 128)]
)
@pytest.mark.parametrize("pin_memory", [False, True])
def test_gpu_fused_matches_torch(dtype, batch_size, vocab, window, pin_memory):
    if not torch.cuda.is_available():
        pytest.skip("requires a GPU")
    rng = random.Random(71)
    config = WindowedPenaltyConfig.parse(
        {
            "window": window,
            "repetition": 1.2,
            "frequency": 0.1,
            "presence": -0.2,
            "exempt_token_ids": [0],
        }
    )
    requests = {
        row: WindowedPenaltyState(
            config, [rng.randrange(vocab) for _ in range(window * 2)]
        )
        for row in range(batch_size)
    }
    batch = SparsePenaltyBatch(torch.device("cuda"), pin_memory)
    batch.update(requests)
    logits = torch.randn(batch_size, vocab, device="cuda", dtype=dtype)
    logits[:, 0] = -float("inf")
    expected = batch.apply_torch(logits.clone())
    actual = batch.apply(logits.clone())
    torch.cuda.synchronize()
    torch.testing.assert_close(
        actual,
        expected,
        rtol=2e-3 if dtype != torch.float32 else 1e-5,
        atol=2e-3 if dtype != torch.float32 else 1e-6,
    )
    assert batch.indices.numel() == 2 * batch.num_updates
    assert batch.parameters.numel() == 2 * batch.num_updates
