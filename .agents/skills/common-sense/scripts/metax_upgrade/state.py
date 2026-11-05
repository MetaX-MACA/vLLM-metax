# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Durable task/contract state; no imports of the adaptation under review."""

import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from datetime import datetime, timezone

if __package__:
    from . import acceptance, workflow, planning, model_coverage
else:
    import acceptance
    import workflow
    import planning
    import model_coverage


SCHEMA_VERSION = 4


KINDS = {
    "static",
    "contract",
    "component_numeric",
    "closure",
    "weight_loading",
    "e2e",
    "batched_test",
    "distributed",
    "graph",
    "review",
}
STATUSES = {"todo", "in_progress", "implemented", "blocked", "done"}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path):
    path = Path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".write-")
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    atomic_write(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")


@contextlib.contextmanager
def locked(run_dir):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / ".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def load(run_dir):
    state = json.loads((Path(run_dir) / "state.json").read_text())
    if state.get("schema_version") not in {1, 2, 3, SCHEMA_VERSION}:
        raise ValueError(
            "Unsupported state schema; preserve the run before migrating it"
        )
    state["run_dir"] = str(Path(run_dir).resolve())
    return state


def save(run_dir, state, *, render_todo=True):
    if render_todo:
        workflow.synchronize(state)
    state["revision"] += 1
    state["updated_at"] = now()
    # State is authoritative. A crash before TODO replacement is repaired on status/resume.
    write_json(Path(run_dir) / "state.json", state)
    if render_todo:
        atomic_write(Path(run_dir) / "TODO.md", render(state))
        if "inventory" in state:
            write_json(Path(run_dir) / "FILES.json", state["inventory"])
            atomic_write(Path(run_dir) / "FILES.md", render_inventory(state))
        if "model_inventory" in state:
            write_json(Path(run_dir) / "MODELS.json", state["model_inventory"])


def relative_file(state, path):
    root = Path(state["inputs"]["metax_source"])
    resolved = (root / path).resolve()
    relative = resolved.relative_to(root)
    if ".git" in relative.parts:
        raise ValueError("Task files must not include Git internals")
    return str(relative)


def source_manifest(state):
    root = Path(state["inputs"]["metax_source"])
    paths = set()
    # Production source, ABI/build definitions and requirements all affect readiness.
    for name in ("vllm_metax", "csrc", "requirements", "cmake"):
        folder = root / name
        if folder.is_dir():
            paths.update(p for p in folder.rglob("*") if p.is_file())
    paths.update(root / p for p in ("setup.py", "pyproject.toml", "CMakeLists.txt"))
    paths.update(
        root / p for task in state.get("tasks", {}).values() for p in task["files"]
    )
    return {
        str(path.relative_to(root)): file_hash(path)
        for path in sorted(paths)
        if "__pycache__" not in path.parts and not path.name.endswith((".pyc", ".pyo"))
    }


def relevant_contracts(state, task_id):
    return {
        key: value
        for key, value in state["contracts"].items()
        if task_id == value.get("producer") or task_id in value.get("consumers", [])
    }


def source_signature(state, task_id):
    task = state["tasks"][task_id]
    root = Path(state["inputs"]["metax_source"])
    return digest(
        {
            "files": {p: file_hash(root / p) for p in task["files"]},
            "configuration": task["configuration"],
            "contracts": relevant_contracts(state, task_id),
            "mapping": task.get("upstream_mapping"),
            "numeric": task.get("numerical_contract"),
            "checks": task.get("local_checks"),
            "model_review": task.get("model_review"),
            "model_review_files": {
                path: file_hash(root / path)
                for path in {
                    **task.get("model_review", {}).get("imports", {}),
                    **task.get("model_review", {}).get("specs", {}),
                }
            },
            "policy": state.get("policy_identity"),
        }
    )


def task_signature(state, task_id):
    task = state["tasks"][task_id]
    return digest(
        {
            "source": source_signature(state, task_id),
            "requirements": task["requires"],
            "acceptance": acceptance.task_identity(state, task_id),
        }
    )


def check_signature(state, task_id, kind, scope=None):
    if kind == "closure" and scope == "model":
        return task_signature(state, task_id)
    return (
        source_signature(state, task_id)
        if kind in workflow.LOCAL_KINDS
        else task_signature(state, task_id)
    )


def validation_identity(state):
    return {
        "source": digest(
            {
                "metax": source_manifest(state),
                "vllm": state["environment"]["fingerprint"],
            }
        ),
        "environment": state["environment"]["fingerprint"],
    }


def effective_requirements(state, task_id):
    waived = planning.deferred_kinds(state, task_id) | model_coverage.waived_kinds(
        state, task_id
    )
    return set(state["tasks"][task_id]["requires"]) - waived


def dependents(state, changed):
    affected = set(changed)
    while True:
        more = {
            key
            for key, task in state["tasks"].items()
            if set(task["depends_on"]) & affected
        }
        for contract in state["contracts"].values():
            if contract.get("producer") in affected:
                more.update(contract.get("consumers", []))
        if more <= affected:
            return affected
        affected.update(more)


def invalidate(state, task_ids, reason, kinds=None):
    for key in dependents(state, task_ids):
        task = state["tasks"][key]
        touched = False
        for evidence in task["evidence"]:
            if (
                kinds is not None
                and evidence["kind"] not in kinds
                and not (
                    evidence["kind"] == "closure"
                    and evidence.get("execution_scope") == "model"
                )
            ):
                continue
            touched = True
            evidence["stale"] = True
            evidence["stale_reason"] = reason
        if task["status"] == "done" and (
            kinds is None or touched or set(task["requires"]) & kinds
        ):
            task["status"] = "implemented"
        task["revalidation_reason"] = reason
    state["completed_at"] = None


def synchronize(state, recover=False):
    current = source_manifest(state)
    previous = state.get("source_manifest", {})
    changed = {
        p for p in previous.keys() | current.keys() if previous.get(p) != current.get(p)
    }
    if changed:
        directly_affected = {
            key for key, task in state["tasks"].items() if set(task["files"]) & changed
        }
        uncovered = changed - {
            p for task in state["tasks"].values() for p in task["files"]
        }
        if uncovered:
            # An unassigned shared helper may affect every task; retain conservative evidence.
            directly_affected.update(state["tasks"])
        invalidate(
            state, directly_affected, "Source changed: " + ", ".join(sorted(changed))
        )
        state["unassigned_changes"] = sorted(
            set(state.get("unassigned_changes", [])) | uncovered
        )
    state["source_manifest"] = current
    workflow.synchronize(state)
    for key, task in state["tasks"].items():
        for evidence in task["evidence"]:
            if not evidence.get("stale") and not evidence_inputs_current(evidence):
                invalidate(state, {key}, "Verification input changed or is missing")
                break
    if recover:
        for task in state["tasks"].values():
            if task["status"] == "in_progress":
                task["status"] = "todo"
                task["notes"].append(
                    {
                        "at": now(),
                        "text": "Interrupted work: inspect existing diff before continuing",
                    }
                )
    return changed


def add_task(
    state, task_id, title, files, dependencies, requires, configuration, next_action
):
    if task_id in state["tasks"]:
        raise ValueError("Task already exists")
    if not requires or not set(requires) <= KINDS:
        raise ValueError("Declare at least one supported evidence kind")
    if task_id in dependencies or not set(dependencies) <= state["tasks"].keys():
        raise ValueError(
            "Dependencies must be existing tasks; combine cycles into one task"
        )
    paths = sorted({relative_file(state, p) for p in files})
    state["tasks"][task_id] = {
        "title": title,
        "files": paths,
        "depends_on": sorted(set(dependencies)),
        "requires": sorted(set(requires)),
        "configuration": configuration,
        "status": "todo",
        "next_action": next_action,
        "notes": [],
        "evidence": [],
    }
    state["unassigned_changes"] = [
        p for p in state.get("unassigned_changes", []) if p not in paths
    ]
    state["completed_at"] = None


def set_scope(state, scope):
    scope = scope.strip()
    if not scope:
        raise ValueError("Scope must be nonempty")
    previous = state["inputs"].get("scope", "")
    if scope == previous:
        return
    state.setdefault("scope_history", []).append(
        {"at": now(), "previous": previous, "current": scope}
    )
    state["inputs"]["scope"] = scope
    invalidate(state, set(state["tasks"]), "Upgrade scope changed")


def missing_evidence(state, task_id, *, check_inputs=True):
    task = state["tasks"][task_id]
    # The latest observation for each kind wins; an earlier pass cannot hide a failure.
    latest = {}
    for evidence in task["evidence"]:
        if evidence["kind"] == "closure" and evidence.get(
            "execution_scope", "component"
        ) != ("model" if task.get("phase") == "validation" else "component"):
            continue
        if not evidence.get("stale"):
            latest[evidence["kind"]] = evidence
    return [
        kind
        for kind in sorted(effective_requirements(state, task_id))
        if kind not in latest
        or latest[kind]["result"] != "passed"
        or latest[kind]["signature"]
        != check_signature(state, task_id, kind, latest[kind].get("execution_scope"))
        or latest[kind]["environment"] != state["environment"]["fingerprint"]
        or (check_inputs and not evidence_inputs_current(latest[kind]))
    ]


def evidence_inputs_current(evidence):
    if "input_hashes" not in evidence:
        return False
    if __package__:
        from . import lifecycle
    else:
        import lifecycle
    try:
        return evidence["input_hashes"] == lifecycle.evidence_inputs(
            evidence["command"], evidence["cwd"], evidence.get("input_files", [])
        )
    except (OSError, ValueError):
        return False


def set_task(state, task_id, status, note, next_action):
    task = state["tasks"][task_id]
    if status not in STATUSES:
        raise ValueError("Unknown task status")
    if status == "done":
        issues = acceptance.validate_task(state, task_id)
        if issues:
            raise ValueError("Task acceptance incomplete: " + "; ".join(issues))
        unfinished = [
            key for key in task["depends_on"] if state["tasks"][key]["status"] != "done"
        ]
        if unfinished:
            raise ValueError("Dependencies are not done: " + ", ".join(unfinished))
        missing = missing_evidence(state, task_id)
        if missing:
            raise ValueError("Missing current passing evidence: " + ", ".join(missing))
    if status == "blocked" and not note:
        raise ValueError("Blocked tasks need a concrete blocker in --note")
    if task["status"] == "done" and status != "done":
        invalidate(state, {task_id}, "Task reopened: " + task_id)
    task["status"] = status
    if note:
        task["notes"].append({"at": now(), "text": note})
    if next_action:
        task["next_action"] = next_action
    state["completed_at"] = None


def put_contract(state, contract):
    required = {"id", "upstream", "summary", "invariants", "producer", "consumers"}
    if not required <= contract.keys() or any(not contract[k] for k in required):
        raise ValueError(
            "Contract needs id, upstream, summary, invariants, producer, consumers"
        )
    participants = {contract["producer"], *contract["consumers"]}
    if not participants <= state["tasks"].keys():
        raise ValueError("Contract participants must be registered tasks")
    if state["contracts"].get(contract["id"]) != contract:
        old = state["contracts"].get(contract["id"], {})
        participants.update([old.get("producer"), *old.get("consumers", [])])
        participants.discard(None)
        state["contracts"][contract["id"]] = contract
        invalidate(state, participants, "Contract updated: " + contract["id"])


def render(state):
    # CLI synchronization validates inputs. Rendering the derived view must not
    # repeatedly hash checkpoint shards; acceptance still checks inputs directly.
    env = state["environment"]
    approval = env.get("confirmation")
    prep_path = (
        Path(state["run_dir"]) / "PREPARATION.json" if state.get("run_dir") else None
    )
    try:
        prep = (
            json.loads(prep_path.read_text())
            if prep_path and prep_path.exists()
            else {}
        )
    except (OSError, ValueError):
        prep = {"status": "invalid preparation record"}
    needs_preparation = (
        bool(prep)
        or state["inputs"]["mode"] == "full"
        or state["inputs"].get("environment_protocol") == "preparation-plan-v1"
    )
    approval_label = (
        "confirmed"
        if approval
        else ("pending preparation" if needs_preparation else "WAITING FOR USER")
    )
    counts = {
        status: sum(t["status"] == status for t in state["tasks"].values())
        for status in sorted(STATUSES)
    }
    lines = [
        "# MetaX upgrade TODO",
        "",
        "Generated from state.json; use the CLI to update progress.",
        "",
        f"- Scope: {state['inputs']['scope']}",
        f"- Mode: {state['inputs']['mode']}",
        f"- State revision: {state['revision']}",
        f"- Initialization: {state.get('initialization', {}).get('phase', 'legacy')}",
        f"- Policy: {'REVIEW REQUIRED' if state.get('policy_change') else 'current'}",
        "- Progress: "
        + ", ".join(f"{status}={count}" for status, count in counts.items()),
        f"- Environment fingerprint: `{env['fingerprint']}`",
        f"- Environment approval: {approval_label}",
        f"- Preparation: {prep.get('status', 'not recorded')}",
        f"- Preparation blocker: {prep.get('blocker', 'none')}",
        f"- Snapshot: `{env['snapshot']}`",
        "- Readable report: `ENVIRONMENT.txt` (available after collection)",
        f"- Completed: {state.get('completed_at') or 'no'}",
        "",
    ]
    rows = state.get("support_matrix", {})
    if "workflow" in state:
        lines.extend(
            [
                f"- Adaptation phase: {state['workflow']['phase']}",
                "- File inventory: [FILES.md](FILES.md) / FILES.json",
                "- Unverified hardware coverage: "
                + str(len(state["workflow"]["unverified"])),
                "",
            ]
        )
        for row in state["workflow"]["unverified"]:
            lines.append(
                f"- UNVERIFIED {row['feature']} on {row['chip']}: {row['reason']}; resume when {row['resume_when']}"
            )
        for key, sample in state.get("model_inventory", {}).items():
            lines.append(
                f"- Model sample **{key}** [{sample['status']}]: rows={', '.join(sample['matrix_rows']) or 'UNBOUND'}"
            )
            if sample.get("skip"):
                label = (
                    "SKIPPED"
                    if model_coverage.valid_skip(state, key)
                    else "SKIP NEEDS RECHECK"
                )
                lines.append(
                    f"  - {label} {sample['skip']['reason_code']}: {sample['skip']['note']}; required={sample['skip']['required']}, available={sample['skip']['available']}"
                )
    lines.extend([f"- Support matrix rows: {len(rows)}", ""])
    for row_id, row in rows.items():
        lines.append(
            f"- Matrix **{row_id}** [{row['status']}]: "
            + json.dumps(row["configuration"], sort_keys=True)
        )
        lines.append(f"  - Tasks: {', '.join(row['tasks'])}")
        lines.append(f"  - Required: {', '.join(row['requires'])}")
    if state.get("policy_change"):
        lines.extend(
            [
                "",
                "Tool or acceptance rules changed; review and run accept-policy-change --note TEXT.",
            ]
        )
    for key, task in state["tasks"].items():
        marker = "x" if task["status"] == "done" else " "
        lines.append(f"- [{marker}] **{key}** [{task['status']}] {task['title']}")
        lines.append(f"  - Configuration: {task['configuration']}")
        if "phase" in task:
            lines.append(f"  - Phase / owner: {task['phase']} / {task['owner']}")
            lines.append(
                f"  - Source adapted, final evidence separate: {workflow.source_ready(state, key)}"
            )
        lines.append(f"  - Depends on: {', '.join(task['depends_on']) or 'none'}")
        lines.append(
            f"  - Pending evidence: {', '.join(missing_evidence(state, key, check_inputs=False)) or 'none'}"
        )
        lines.append(f"  - Next: {task['next_action']}")
        if task["notes"]:
            lines.append(f"  - Latest note: {task['notes'][-1]['text']}")
        if task.get("revalidation_reason"):
            lines.append(f"  - Revalidation: {task['revalidation_reason']}")
    if not state["tasks"]:
        lines.append(
            "No tasks yet. Inventory all production modules and create the dependency worklist."
        )
    if state.get("unassigned_changes"):
        lines.extend(["", "Unassigned source changes (must assign before finish):"])
        lines.extend(f"- `{p}`" for p in state["unassigned_changes"])
    if state["checkpoints"]:
        last = state["checkpoints"][-1]
        lines.extend(
            ["", "Latest handoff:", last["note"], "", "Next: " + last["next_action"]]
        )
    return "\n".join(lines) + "\n"


def render_inventory(state):
    lines = [
        "# Full file inventory",
        "",
        "Generated from state.json. Completion is scoped to the declared hardware/configurations; unverified branches remain listed in TODO.md.",
        "",
        "| File | Status | Owner skill | Phase | Source ready | Disposition |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for path, row in sorted(state["inventory"].items()):
        owners = ", ".join(row["owners"]) or "unassigned"
        dispositions = ", ".join(row["dispositions"]) or "pending review"
        source = "yes; final evidence separate" if row["source_ready"] else "no"
        lines.append(
            f"| `{path}` | {row['label']} | {owners} | {row['phase']} | {source} | {dispositions} |"
        )
    return "\n".join(lines) + "\n"
