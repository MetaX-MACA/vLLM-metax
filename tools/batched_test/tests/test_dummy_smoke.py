# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Dummy execution acceptance is distinct from real-weight semantic accuracy."""

import threading
from types import SimpleNamespace

import pytest

from tools.batched_test import suites
from tools.batched_test.api_client import ChatCompletionClient
from tools.batched_test.model_worker import InferenceTask
from tools.batched_test.reporting import inference_row
from tools.batched_test.results import Phase, Status, TaskOutcome, TaskResult
from tools.batched_test.specs import (
    ModelSpec,
    ServeSpec,
    InferenceSpec,
    SuiteSpec,
    CaseSpec,
)


@pytest.mark.parametrize("kind", ["text", "image"])
def test_smoke_client_requests_decode_tokens_and_records_usage(kind):
    requests = []

    def create(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(
            usage=SimpleNamespace(completion_tokens=2),
            choices=[SimpleNamespace(message=SimpleNamespace(content="generated"))],
        )

    api = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    connection = SimpleNamespace(
        get_model=lambda: "dummy", check=lambda: None, request_client=lambda: api
    )
    client = ChatCompletionClient(connection=connection)
    outputs = (
        client.run_text_only(["prompt"], smoke=True)
        if kind == "text"
        else client.run_single_image(
            2, ["https://example.invalid/image.png"], smoke=True
        )
    )
    assert list(outputs) == ["generated"]
    assert client.last_completion_tokens == 2
    assert requests[0]["extra_body"] == {"ignore_eos": True, "min_tokens": 2}


def test_dummy_load_selects_execution_smoke_and_cannot_claim_accuracy(tmp_path):
    cases = tmp_path / "cases.yaml"
    cases.write_text("- question: question\n  keywords: [semantic-answer]\n")
    model = ModelSpec(
        "dummy",
        str(tmp_path),
        ServeSpec(extra_args=("--load-format", "dummy")),
        infer_types=("text-only",),
    )
    spec = suites.plan_inference(model, text_case=str(cases), image_case=None)
    assert spec.validation_mode == "dummy_smoke"
    with pytest.raises(ValueError, match="Dummy weights"):
        suites.plan_inference(
            model, text_case=str(cases), image_case=None, validation_mode="accuracy"
        )


@pytest.mark.parametrize(
    "content,tokens,passed",
    [("random tokens", 2, True), ("", 2, False), ("one token", 1, False)],
)
def test_smoke_exercises_concurrent_requests_and_decode_without_keyword_scoring(
    tmp_path, monkeypatch, content, tokens, passed
):
    barrier = threading.Barrier(2)
    closed = []
    connection = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(suites, "ApiConnection", lambda **kwargs: connection)

    class Client:
        def __init__(self, **kwargs):
            self.last_completion_tokens = tokens

        def run_text_only(self, questions, max_completion_tokens, smoke=False):
            assert smoke and max_completion_tokens >= 2
            barrier.wait(timeout=3)
            yield content

    monkeypatch.setattr(suites, "ChatCompletionClient", Client)
    spec = InferenceSpec(
        (SuiteSpec("text-only", (CaseSpec("question", ("semantic-answer",), 2),)),),
        validation_mode="dummy_smoke",
    )
    context = SimpleNamespace(
        cancel=threading.Event(),
        request_timeout=3,
        work_dir=tmp_path,
        check=lambda: None,
    )
    cases = suites.run_suites(context, spec, 8000)
    assert len(cases) == 2
    assert all(case.passed == passed for case in cases)
    assert closed == [True]
    model = ModelSpec(
        "dummy", str(tmp_path), ServeSpec(extra_args=("--load-format", "dummy"))
    )
    result = TaskResult(
        Status.PASSED if passed else Status.FAILED,
        Phase.COMPLETE,
        1,
        TaskOutcome(cases=cases),
    )
    row = inference_row(InferenceTask(model, spec), result)
    assert row["Correct Ratio"] == "N/A"
    assert row["Validation Mode"] == "dummy_smoke"
    assert row["Stage"] == ("DUMMY_SMOKE_PASSED" if passed else "DUMMY_SMOKE_FAILED")
