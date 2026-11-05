# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Result reuse must identify code, environment, checkpoint bytes and inputs."""

from dataclasses import replace
import os

from tools.batched_test import identity
from tools.batched_test.model_worker import InferenceTask
from tools.batched_test.reporting import CsvReport, ResumeIndex
from tools.batched_test.results import CaseResult, Phase, Status, TaskOutcome, TaskResult
from tools.batched_test.specs import ModelSpec, ServeSpec, InferenceSpec, SuiteSpec, CaseSpec


def fixture(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"architectures": ["Example"]}')
    (model_dir / "weights.bin").write_bytes(b"original weights")
    template = tmp_path / "chat.jinja"
    template.write_text("original template")
    model = ModelSpec("example", str(model_dir), ServeSpec(extra_args=("--chat-template", str(template))))
    spec = InferenceSpec((SuiteSpec("text-only", (CaseSpec("question", ("answer",)),)),))
    return model, spec, template


def test_resume_rejects_changed_source_environment_weights_and_inputs(tmp_path):
    model, spec, template = fixture(tmp_path)
    runtime = {"source": "source-v1", "environment": "environment-v1"}
    original = InferenceTask(model, spec, validation_identity=identity.capture(model, spec, runtime))
    path = tmp_path / "results.csv"
    result = TaskResult(Status.PASSED, Phase.COMPLETE, 1, TaskOutcome(cases=(CaseResult("text-only", 0, True, "answer"),)))
    with CsvReport(path, [original], "inference") as report:
        report.write(original, result)
    resume = ResumeIndex(str(path))
    assert resume.should_skip(original)
    assert not resume.should_skip(replace(original, validation_identity=identity.capture(model, spec, {**runtime, "source": "source-v2"})))
    assert not resume.should_skip(replace(original, validation_identity=identity.capture(model, spec, {**runtime, "environment": "environment-v2"})))
    weights = tmp_path / "model/weights.bin"
    before = weights.stat()
    weights.write_bytes(b"modified weights")
    os.utime(weights, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert not resume.should_skip(replace(original, validation_identity=identity.capture(model, spec, runtime)))
    weights.write_bytes(b"original weights")
    template.write_text("modified template")
    assert not resume.should_skip(replace(original, validation_identity=identity.capture(model, spec, runtime)))


def test_code_content_identity_is_independent_of_mtime(tmp_path, monkeypatch):
    monkeypatch.delenv("METAX_UPGRADE_VALIDATION_IDENTITY_FILE", raising=False)
    code = tmp_path / "vllm_metax/op.py"
    code.parent.mkdir()
    code.write_text("value = 1\n")
    before = code.stat()
    first = identity.runtime_identity([tmp_path])
    code.write_text("value = 2\n")
    os.utime(code, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert identity.runtime_identity([tmp_path])["source"] != first["source"]


def test_unbound_and_mutable_remote_results_are_not_reused(tmp_path):
    model, spec, _ = fixture(tmp_path)
    spec = InferenceSpec((SuiteSpec("single-image", (CaseSpec("https://example.invalid/mutable.png", ("answer",)),)),))
    value = identity.capture(model, spec, {"source": "s", "environment": "e"})
    assert not value.reusable
    remote = replace(model, model_path="organization/model")
    assert not identity.capture(remote, spec, {"source": "s", "environment": "e"}).reusable
