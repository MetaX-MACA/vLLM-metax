# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Task implementations. Resource lifetime belongs exclusively to TaskRunner."""

from dataclasses import asdict, dataclass
import hashlib
import json
import os
import regex as re
import shlex

from tools.batched_test import deployment
from tools.batched_test.results import Phase, Status, TaskOutcome
from tools.batched_test.runtime import RunContext
from tools.batched_test.specs import BenchmarkSpec, InferenceSpec, ModelSpec
from tools.batched_test.suites import run_suites


def task_fingerprint(model, spec) -> str:
    def canonical(value):
        if isinstance(value, dict):
            return {key: canonical(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [canonical(item) for item in value]
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return value

    payload = json.dumps(
        canonical({"model": asdict(model), "spec": asdict(spec)}), sort_keys=True
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def artifact_id(model: ModelSpec, spec: InferenceSpec, kind: str):
    tag = re.sub(r"[^a-zA-Z0-9_.-]", "_", model.tag(kind))
    return f"{tag}_{task_fingerprint(model, spec)[:12]}"


@dataclass(frozen=True)
class InferenceTask:
    # Immutable model/service settings; construction performs no resource allocation.
    model: ModelSpec
    # Validated suites and accuracy policy to execute against the service.
    spec: InferenceSpec
    # Optional disambiguated CSV name; excluded from the configuration fingerprint.
    report_name: str | None = None
    # Report category and output subtree for correctness tests.
    kind = "inference"

    @property
    def report_tag(self):
        return self.report_name or self.model.tag(self.kind)

    @property
    def fingerprint(self):
        return task_fingerprint(self.model, self.spec)

    @property
    def artifact_id(self):
        return artifact_id(self.model, self.spec, self.kind)

    def execute(self, context: RunContext) -> TaskOutcome:
        context.phase = Phase.STARTING
        plan = deployment.plan_deployment(context)
        process = deployment.start_service(context, plan)
        deployment.wait_ready(context, plan, process)
        context.phase = Phase.TESTING
        cases = run_suites(context, self.spec, plan.port)
        ratio = sum(case.passed for case in cases) / len(cases) if cases else 0.0
        passed = bool(cases) and ratio >= self.spec.pass_threshold
        return TaskOutcome(
            Status.PASSED if passed else Status.FAILED,
            cases,
            ""
            if passed
            else f"Correct ratio below required {self.spec.pass_threshold:.0%}",
            {"log_dir": str(context.work_dir)},
        )


@dataclass(frozen=True)
class BenchmarkTask:
    # Immutable model/service settings used to construct the sweep command.
    model: ModelSpec
    # Frozen parameter combinations, dataset settings and repetition counts.
    spec: BenchmarkSpec
    # Optional disambiguated CSV name; excluded from the configuration fingerprint.
    report_name: str | None = None
    # Report category and output subtree for performance sweeps.
    kind = "performance"

    @property
    def report_tag(self):
        return self.report_name or self.model.tag(self.kind)

    @property
    def fingerprint(self):
        return task_fingerprint(self.model, self.spec)

    @property
    def artifact_id(self):
        return artifact_id(self.model, self.spec, self.kind)

    def execute(self, context: RunContext) -> TaskOutcome:
        context.phase = Phase.STARTING
        plan = deployment.plan_deployment(context)
        rank0 = plan.ranks[0]
        bench = deployment.bench_command(self.model, self.spec, plan.port)
        params = context.work_dir / "bench_params.json"
        params.write_text(json.dumps(self.spec.parameter_dicts()), encoding="utf-8")
        command = deployment.sweep_command(
            list(rank0.command),
            bench,
            self.spec,
            params,
            context.work_dir / "sweep",
        )
        log = str(context.work_dir / "sweep.log")
        env = {**os.environ, **dict(rank0.env)}
        deployment.log_command(log, command, dict(rank0.env))
        # Sweep owns rank0's service; do not launch another server here.
        process = context.session.start(command, env, log)
        deployment.launch_remotes(context, plan)
        context.phase = Phase.TESTING
        code = context.session.wait(process)
        if code != 0:
            raise RuntimeError(f"vllm bench sweep exited with code {code}; see {log}")
        return TaskOutcome(
            artifacts={
                "log_dir": log,
                "server_command": shlex.join(rank0.command),
                "client_command": deployment.client_commands(bench, self.spec),
                "env": dict(rank0.env),
            }
        )
