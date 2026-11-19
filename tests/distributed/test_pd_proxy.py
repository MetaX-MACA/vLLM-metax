# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import pytest

from vllm_metax.distributed import pd_proxy


def _producer_response():
    return {
        "choices": [{"text": "x"}],
        "kv_transfer_params": {
            "do_remote_prefill": True,
            "do_remote_decode": False,
            "remote_engine_id": "engine-p",
            "remote_request_id": "request-p",
            "remote_host": "127.0.0.1",
            "remote_port": 12345,
            "tp_size": 1,
            "dcp_size": 1,
            "pp_size": 1,
            "remote_num_tokens": 4,
            "remote_block_ids": [[1, 2]],
            "transfer_mode": "pull",
        },
    }


def test_prefill_request_forces_one_token_and_preserves_prompt():
    payload = {
        "prompt": [1, 2, 3],
        "n": 1,
        "max_tokens": 32,
        "max_completion_tokens": 40,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.2,
    }
    result = pd_proxy.prefill_request(payload)
    assert result["prompt"] == [1, 2, 3]
    assert result["max_tokens"] == 1
    assert result["stream"] is False
    assert result["temperature"] == 0.2
    assert result["kv_transfer_params"] == {"do_remote_decode": True}
    assert "max_completion_tokens" not in result
    assert "stream_options" not in result


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": [1, 2], "n": 2},
        {"prompt": [[1, 2], [3, 4]], "n": 1},
        {"prompt": [1, 2], "n": 1, "kv_transfer_params": {"do_remote_decode": True}},
    ],
)
def test_prefill_request_rejects_unsupported_shape(payload):
    with pytest.raises(ValueError):
        pd_proxy.prefill_request(payload)


def test_decode_request_preserves_original_payload_and_handoff():
    payload = {"prompt": [1, 2, 3], "max_tokens": 8, "temperature": 0.0}
    result = pd_proxy.decode_request(payload, _producer_response())
    assert result["prompt"] == payload["prompt"]
    assert result["max_tokens"] == 8
    assert result["kv_transfer_params"] == _producer_response()["kv_transfer_params"]


@pytest.mark.parametrize(
    "mutator",
    [
        lambda response: response["kv_transfer_params"].update(do_remote_prefill=False),
        lambda response: response["kv_transfer_params"].update(remote_host=""),
        lambda response: response["kv_transfer_params"].update(remote_port=70000),
        lambda response: response["kv_transfer_params"].update(pp_size=2),
        lambda response: response["kv_transfer_params"].update(remote_block_ids=[]),
        lambda response: response["kv_transfer_params"].update(transfer_mode="push"),
    ],
)
def test_decode_request_rejects_invalid_handoff(mutator):
    response = _producer_response()
    mutator(response)
    with pytest.raises(ValueError):
        pd_proxy.decode_request({"prompt": [1, 2]}, response)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("http://host:8000/", ("http://host:8000",)),
        (
            ["http://p1:8000", "https://p2:8443/base"],
            ("http://p1:8000", "https://p2:8443/base"),
        ),
    ],
)
def test_normalize_urls_accepts_http_pools(value, expected):
    assert pd_proxy.normalize_urls(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        [],
        "ftp://host:8000",
        "http://user:pass@host:8000",
        "http://host:8000?query=1",
        "http://host:8000#fragment",
        "http://host:bad",
        "http://host:0",
    ],
)
def test_normalize_urls_rejects_unsafe_or_malformed_urls(value):
    with pytest.raises(ValueError):
        pd_proxy.normalize_urls(value)
