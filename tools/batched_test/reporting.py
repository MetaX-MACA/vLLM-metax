# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""CSV compatibility, resume lookup and configuration fingerprints."""

import csv
from dataclasses import asdict
import json
from pathlib import Path

from tools.batched_test.results import Phase, Status, TaskResult

INFERENCE_FIELDS = ["Model", "Correct Ratio", "Stage", "Reason", "Model Path"]
BENCHMARK_FIELDS = [
    "task_name",
    "status",
    "log_dir",
    "error",
    "server_command",
    "client_command",
    "env",
]


def inference_row(task, result: TaskResult) -> dict:
    stage = {
        Status.PASSED: "NORMAL_END",
        Status.SKIPPED: "NORMAL_END",
        Status.FAILED: "ACCURACY_FAILED",
        Status.TIMEOUT: "TIMEOUT",
        Status.CANCELLED: "CANCELLED",
    }.get(
        result.status,
        {
            Phase.ALLOCATING: "INIT",
            Phase.STARTING: "STARTING_SERVER",
            Phase.TESTING: "INFERENCING",
            Phase.CLEANUP: "CRASH",
        }.get(result.phase, "CRASH"),
    )
    reason = " | ".join(
        filter(None, (result.error, result.outcome.reason, result.cleanup_error))
    )
    ratio = 1.0 if result.status == Status.SKIPPED else result.outcome.correct_ratio
    return {
        "Model": task.report_tag,
        "Correct Ratio": f"{ratio * 100:g}%",
        "Stage": stage,
        "Reason": reason.replace("\n", " | ")[:1000],
        "Model Path": task.model.model_path,
    }


def benchmark_row(task, result: TaskResult) -> dict:
    artifacts = result.outcome.artifacts
    return {
        "task_name": task.report_tag,
        "status": {Status.PASSED: "success", Status.TIMEOUT: "timeout"}.get(
            result.status, "error"
        ),
        "log_dir": artifacts.get("log_dir"),
        "error": " | ".join(filter(None, (result.error, result.cleanup_error))) or None,
        "server_command": {"type": "normal", "command": artifacts["server_command"]}
        if "server_command" in artifacts
        else None,
        "client_command": artifacts.get("client_command"),
        "env": {"type": "normal", "server_cmd_env": artifacts["env"]}
        if "env" in artifacts
        else None,
    }


def manifest_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".manifest.json")


class ResumeIndex:
    def __init__(self, path: str | None):
        # Previous CSV rows grouped by report tag for successful-result matching.
        self.rows: dict[str, list[dict[str, str]]] = {}
        # Optional tag-to-fingerprint manifest; None denotes a legacy CSV without one.
        self.fingerprints: dict[str, str] | None = None
        if path:
            with open(path, encoding="utf-8") as stream:
                for row in csv.DictReader(stream):
                    self.rows.setdefault(row["Model"].strip(), []).append(row)
            manifest = manifest_path(Path(path))
            if manifest.exists():
                self.fingerprints = json.loads(manifest.read_text(encoding="utf-8"))

    def should_skip(self, task) -> bool:
        tag = task.report_tag
        rows = self.rows.get(tag, [])
        if not rows:
            return False
        if (
            self.fingerprints is not None
            and self.fingerprints.get(tag) != task.fingerprint
        ):
            return False
        for row in rows:
            try:
                ratio = float(row["Correct Ratio"].strip().rstrip("%"))
            except (ValueError, KeyError):
                return False
            if (
                row.get("Stage", "").strip() != "NORMAL_END"
                or ratio != 100
                or row.get("Model Path", "") != task.model.model_path
            ):
                return False
        return True


class CsvReport:
    def __init__(self, path: Path, tasks, kind: str):
        # Absolute CSV output path; the manifest is written beside it.
        self.path = path
        # Planned tasks whose fingerprints populate the report manifest.
        self.tasks = tasks
        # Selects inference or performance CSV schema and row conversion.
        self.kind = kind

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path(self.path).write_text(
            json.dumps(
                {task.report_tag: task.fingerprint for task in self.tasks}, indent=2
            ),
            encoding="utf-8",
        )
        # CSV file handle opened on context entry and closed on exit.
        self.stream = self.path.open("w", newline="", encoding="utf-8")
        # Schema-specific DictWriter bound to stream while the report is open.
        self.writer = csv.DictWriter(
            self.stream,
            fieldnames=(
                INFERENCE_FIELDS if self.kind == "inference" else BENCHMARK_FIELDS
            ),
        )
        self.writer.writeheader()
        self.stream.flush()
        return self

    def write(self, task, result):
        artifact_dir = result.outcome.artifacts.get("artifact_dir")
        if artifact_dir:
            (Path(artifact_dir) / "result.json").write_text(
                json.dumps(asdict(result), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        row = (
            inference_row(task, result)
            if self.kind == "inference"
            else benchmark_row(task, result)
        )
        self.writer.writerow(row)
        self.stream.flush()
        return row

    def __exit__(self, *_):
        self.stream.close()
