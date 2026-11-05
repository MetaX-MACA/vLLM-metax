# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Baseline model coverage, concrete-import audit and explicit resource skips."""

import json
from pathlib import Path
import subprocess
import sys

MODEL_KINDS = {"e2e", "batched_test", "weight_loading", "distributed", "graph"}


def _store():
    if __package__:
        from . import state
    else:
        import state
    return state


def synchronize(state):
    root = Path(state["inputs"]["metax_source"])
    models = state.setdefault("model_inventory", {})
    for path in sorted((root / "tools/models").rglob("config.json")):
        key = str(path.parent.relative_to(root))
        data = json.loads(path.read_text())
        row = models.setdefault(key, {"status": "pending", "matrix_rows": []})
        row.update(
            config=str(path.relative_to(root)),
            config_hash=_store().file_hash(path),
            architectures=data.get("architectures", []),
        )
    for key, row in models.items():
        row["config_hash"] = _store().file_hash(root / row["config"])
        if row.get("verified_signature") and row[
            "verified_signature"
        ] != sample_signature(state, key):
            row["status"] = "pending"
        if row.get("skip") and not valid_skip(state, key):
            row["status"] = "pending"


def review_model(run_dir, state, task_id, entries, specs):
    store = _store()
    task = state["tasks"][task_id]
    root = Path(state["inputs"]["metax_source"])
    entries = [store.relative_file(state, value) for value in entries]
    specs = [store.relative_file(state, value) for value in specs]
    if not entries or not specs or not set(entries + specs) <= set(task["files"]):
        raise ValueError("Model entries and specs must be owned task files")
    if any(
        not path.startswith("vllm_metax/models/")
        or not (root / path).is_file()
        or not (root / path).read_text().strip()
        for path in specs
    ):
        raise ValueError("Every model needs a nonempty spec in its model directory")
    auditor = (
        Path(__file__).resolve().parents[3]
        / "model-adaptation/scripts/audit_imports.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(auditor),
            "--source",
            str(root),
            *[arg for entry in entries for arg in ("--entry", entry)],
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode:
        raise ValueError(
            "Model import audit failed: " + (result.stdout or result.stderr)
        )
    audit = json.loads(result.stdout)
    record = {
        "entries": entries,
        "specs": {path: store.file_hash(root / path) for path in specs},
        "imports": audit["files"],
        "audit": audit,
    }
    artifact = "model-reviews/" + store.digest(record) + ".json"
    store.write_json(Path(run_dir) / artifact, record)
    store.invalidate(state, {task_id}, "Model spec/import audit updated")
    task["model_review"] = {
        "artifact": artifact,
        "sha256": store.file_hash(Path(run_dir) / artifact),
        **record,
    }


def review_issues(state, task_id):
    task = state["tasks"][task_id]
    if task.get("phase") != "models" or not task["files"]:
        return []
    record = task.get("model_review")
    if not record:
        return [
            "Record each model spec and concrete-model import audit with model-review"
        ]
    store = _store()
    root = Path(state["inputs"]["metax_source"])
    issues = []
    for path, content_hash in {**record["specs"], **record["imports"]}.items():
        if store.file_hash(root / path) != content_hash:
            issues.append("Model spec/import closure changed: " + path)
    if store.file_hash(Path(state["run_dir"]) / record["artifact"]) != record["sha256"]:
        issues.append("Model audit artifact is missing or changed")
    return issues


def bind(state, document):
    store = _store()
    root = Path(state["inputs"]["metax_source"])
    rows = document.get("models")
    if not isinstance(rows, list):
        raise ValueError("Model binding needs a models list")
    for binding in rows:
        key = binding.get("id")
        if key not in state["model_inventory"]:
            raise ValueError("Unknown baseline model sample: " + str(key))
        spec = store.relative_file(state, binding.get("spec", ""))
        if not spec.startswith("vllm_metax/models/") or not (root / spec).is_file():
            raise ValueError("Baseline model needs its local model-directory spec")
        matrix = binding.get("matrix_rows")
        if not isinstance(matrix, list) or not matrix:
            raise ValueError("Bind every model to its intended support-matrix rows")
        for row_id in matrix:
            row = state["support_matrix"].get(row_id)
            if not row or row.get("model_sample") != key:
                raise ValueError("Support row must identify the matching model_sample")
            model_path = row["configuration"].get("model_path")
            mode = row["configuration"].get("validation_mode")
            if mode not in {"accuracy", "dummy_smoke"} or (
                row["configuration"].get("load_format") == "dummy"
                and mode != "dummy_smoke"
            ):
                raise ValueError(
                    "Model rows must distinguish real-weight accuracy from dummy_smoke"
                )
            if (
                not isinstance(model_path, str)
                or (root / model_path).resolve() != (root / key).resolve()
            ):
                raise ValueError(
                    "Support row must execute the actual baseline model directory"
                )
            if not any(
                spec in state["tasks"][task].get("model_review", {}).get("specs", {})
                for task in row["tasks"]
            ):
                raise ValueError(
                    "Bind a source task with a current model-review/spec to each model row"
                )
        state["model_inventory"][key].update(
            spec=spec, matrix_rows=matrix, status="pending"
        )
        state["model_inventory"][key].pop("skip", None)
    state["completed_at"] = None


def sample_signature(state, key):
    store = _store()
    sample = state["model_inventory"][key]
    tasks = {
        task
        for row in sample["matrix_rows"]
        for task in state["support_matrix"].get(row, {}).get("tasks", [])
    }
    return store.digest(
        {
            "config": sample["config_hash"],
            "environment": state["environment"]["fingerprint"],
            "sources": {
                task: store.source_signature(state, task) for task in sorted(tasks)
            },
            "matrix": {
                row: state["support_matrix"].get(row) for row in sample["matrix_rows"]
            },
        }
    )


def skip(state, key, reason_code, required, available, evidence_file, note):
    store = _store()
    sample = state["model_inventory"].get(key)
    if not sample or not sample.get("matrix_rows"):
        raise ValueError("Bind the baseline model before recording a resource skip")
    if state.get("workflow", {}).get("phase") != "validation":
        raise ValueError("Record model resource skips only in final validation")
    if (
        reason_code not in {"insufficient_gpus", "insufficient_vram"}
        or required <= available
        or available < 0
        or not note.strip()
    ):
        raise ValueError(
            "Resource skip needs a real GPU-count or VRAM deficit and explanation"
        )
    evidence_file = Path(evidence_file).resolve()
    if not evidence_file.is_file() or not evidence_file.read_text().strip():
        raise ValueError(
            "Resource skip needs saved device/memory evidence or an actual OOM log"
        )
    sample["skip"] = {
        "reason_code": reason_code,
        "required": required,
        "available": available,
        "note": note,
        "evidence_file": str(evidence_file),
        "evidence_hash": store.file_hash(evidence_file),
        "signature": sample_signature(state, key),
    }
    sample["status"] = "skipped"
    state["completed_at"] = None


def valid_skip(state, key):
    sample = state.get("model_inventory", {}).get(key, {})
    record = sample.get("skip")
    return bool(
        record
        and record["signature"] == sample_signature(state, key)
        and _store().file_hash(record["evidence_file"]) == record["evidence_hash"]
    )


def waived_kinds(state, task_id):
    rows = [
        row
        for row in state.get("support_matrix", {}).values()
        if task_id in row["tasks"]
    ]
    return (
        MODEL_KINDS
        if rows
        and all(
            row.get("model_sample") and valid_skip(state, row["model_sample"])
            for row in rows
        )
        else set()
    )


def finish_issues(state):
    if state["inputs"]["mode"] != "full":
        return []
    store = _store()
    errors = []
    for key, sample in state.get("model_inventory", {}).items():
        if not sample["config_hash"]:
            errors.append("Baseline model configuration was removed: " + key)
            continue
        if valid_skip(state, key):
            sample["status"] = "skipped"
            continue
        sample["status"] = "pending"
        if not sample.get("matrix_rows"):
            errors.append("Baseline model is not bound to spec/tests: " + key)
            continue
        for row_id in sample["matrix_rows"]:
            row = state["support_matrix"].get(row_id)
            if (
                not row
                or row.get("model_sample") != key
                or row["status"] != "supported"
            ):
                errors.append(
                    "Baseline model has no supported test configuration: " + key
                )
                continue
            evidence = {
                e["kind"]
                for task in row["tasks"]
                for e in store.acceptance._current_evidence(state, task)
            }
            if not {"e2e", "batched_test"} <= evidence:
                errors.append(
                    "Baseline model needs current e2e and batched inference: " + key
                )
        if not any(key in error for error in errors):
            sample["status"] = "passed"
            sample["verified_signature"] = sample_signature(state, key)
    return errors
