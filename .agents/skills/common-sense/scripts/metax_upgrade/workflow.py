# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""File inventory and ordered source adaptation, independent of candidate imports."""

from pathlib import Path


PHASES = (
    "platform",
    "registry",
    "attention",
    "framework",
    "models",
    "patch",
    "validation",
)
MODEL_KINDS = {"weight_loading", "e2e", "distributed", "graph", "batched_test"}
LOCAL_KINDS = {"static", "review", "contract", "component_numeric", "closure"}
LABELS = {"todo": "未适配", "in_progress": "正在适配中", "done": "适配完成并验证完毕"}
OWNERS = {
    "platform": "common-sense",
    "registry": "registry-upgrade",
    "attention": "attention-upgrade",
    "framework": "common-sense",
    "models": "model-upgrade",
    "patch": "patch-upgrade",
    "validation": "vllm-metax-upgrade",
}


def _store():
    if __package__:
        from . import state
    else:
        import state
    return state


def classify(path):
    if path == "vllm_metax/platform.py":
        return "platform"
    for folder, phase in (
        ("registry", "registry"),
        ("models", "models"),
        ("patch", "patch"),
    ):
        if path.startswith("vllm_metax/" + folder + "/"):
            return phase
    if path.startswith("vllm_metax/v1/attention/") or any(
        part in path for part in ("/layers/attention/", "/kernels/attention/")
    ):
        return "attention"
    return "framework"


def initialize(state):
    state["workflow"] = {"phase": "platform", "history": [], "unverified": []}
    state["inventory"] = {}
    synchronize(state)


def register_task(state, task_id, phase=None, owner=None, exception_note=""):
    task = state["tasks"][task_id]
    phase = phase or max(
        (classify(p) for p in task["files"]), key=PHASES.index, default="validation"
    )
    if phase not in PHASES:
        raise ValueError("Unknown workflow phase")
    for path in task["files"]:
        natural = classify(path)
        if path.startswith(
            ("vllm_metax/", "csrc/", "requirements/", "cmake/")
        ) or path in {"setup.py", "pyproject.toml", "CMakeLists.txt"}:
            if phase == "validation":
                raise ValueError(
                    "Production modules cannot be assigned to the validation phase"
                )
            if natural in {"platform", "models", "patch"} and phase != natural:
                raise ValueError(f"{path} belongs to the {natural} source phase")
            if natural in {"registry", "attention", "framework"} and PHASES.index(
                phase
            ) > PHASES.index(natural):
                raise ValueError(f"{path} must be adapted by the {natural} phase")
    owner = (owner or OWNERS[phase]).strip()
    if not owner:
        raise ValueError("A task needs an owning skill")
    for other_id, other in state["tasks"].items():
        if other_id == task_id or not set(other["files"]) & set(task["files"]):
            continue
        if other.get("needs_replan"):
            continue
        if other.get("phase", phase) != phase or other.get("owner", owner) != owner:
            raise ValueError(
                "Shared files need one phase and owning skill; split consumer validation tasks"
            )
    task.update(phase=phase, owner=owner, exception_note=exception_note.strip())
    synchronize(state)


def source_ready(state, task_id):
    task = state["tasks"][task_id]
    store = _store()
    if task.get("needs_replan"):
        return False
    if task.get("source_signature") != store.source_signature(state, task_id):
        return False
    if store.planning.source_issues(
        state, task_id
    ) or store.model_coverage.review_issues(state, task_id):
        return False
    if set(store.missing_evidence(state, task_id, check_inputs=False)) & LOCAL_KINDS:
        return False
    if (
        state["inputs"]["mode"] == "full"
        and task.get("phase") == "patch"
        and not task.get("final_patch_review")
    ):
        return False
    return task["status"] in {"implemented", "done"}


def synchronize(state):
    if "workflow" not in state:
        return
    store = _store()
    store.model_coverage.synchronize(state)
    root = Path(state["inputs"]["metax_source"])
    inventory = state.setdefault("inventory", {})
    paths = {
        p for p, value in state.get("source_manifest", {}).items() if value is not None
    }
    paths.update(p for task in state["tasks"].values() for p in task["files"])
    for path in sorted(paths):
        inventory.setdefault(
            path,
            {"phase": classify(path), "initial_hash": store.file_hash(root / path)},
        )
    ready = {key: source_ready(state, key) for key in state["tasks"]}
    verified = {}
    for key, task in state["tasks"].items():
        current = {}
        for evidence in task["evidence"]:
            if not evidence.get("stale"):
                current[evidence["kind"]] = evidence
        # Synchronization already checks input identities; rendering must not
        # hash large checkpoint inputs again for every inventory row.
        verified[key] = (
            task["status"] == "done"
            and not store.missing_evidence(state, key, check_inputs=False)
            and all(
                not store.acceptance.validate_report(
                    state, key, kind, evidence.get("report", evidence.get("origins"))
                )
                for kind, evidence in current.items()
                if kind in task["requires"]
            )
        )
    for path, row in inventory.items():
        tasks = [key for key, task in state["tasks"].items() if path in task["files"]]
        row["tasks"] = tasks
        row["current_hash"] = store.file_hash(root / path)
        row["present"] = row["current_hash"] is not None
        row["owners"] = sorted(
            {state["tasks"][key].get("owner", "unassigned") for key in tasks}
        )
        assigned = [key for key in tasks if not state["tasks"][key].get("needs_replan")]
        if assigned:
            row["phase"] = state["tasks"][assigned[0]].get("phase", row["phase"])
        elif tasks:
            row["phase"] = classify(path)
        row["source_ready"] = bool(tasks) and all(ready[key] for key in tasks)
        done = bool(tasks) and all(verified[key] for key in tasks)
        started = any(state["tasks"][key]["status"] != "todo" for key in tasks)
        row["status"] = "done" if done else "in_progress" if started else "todo"
        row["label"] = LABELS[row["status"]]
        row["dispositions"] = sorted(
            {
                item["disposition"]
                for key in tasks
                for item in state["tasks"][key]
                .get("upstream_mapping", {})
                .get("mappings", [])
                if item.get("local") == path and "disposition" in item
            }
        )


def phase_issues(state, phase=None):
    synchronize(state)
    phase = phase or state["workflow"]["phase"]
    return [
        path
        for path, row in state["inventory"].items()
        if PHASES.index(row["phase"]) <= PHASES.index(phase) and not row["source_ready"]
    ]


def assert_task_start(state, task_id):
    if state["inputs"]["mode"] != "full":
        return
    task = state["tasks"][task_id]
    issues = _store().planning.analysis_issues(state)
    if issues:
        raise ValueError("; ".join(issues))
    if task.get("needs_replan"):
        raise ValueError(
            "Legacy task crosses phase boundaries; use task-replan and assign remaining files"
        )
    current = state["workflow"]["phase"]
    phase = task["phase"]
    if PHASES.index(phase) > PHASES.index(current):
        if phase == "patch" and task.get("exception_note"):
            return
        raise ValueError(f"Current phase is {current}; cannot start {phase}")
    if phase == "models":
        pending = phase_issues(state, "framework")
        if pending:
            raise ValueError(
                "Framework source adaptation is incomplete: " + ", ".join(pending)
            )


def mark_implemented(state, task_id, note):
    if state["inputs"]["mode"] != "full":
        state["tasks"][task_id]["source_signature"] = _store().source_signature(
            state, task_id
        )
        return
    store = _store()
    task = state["tasks"][task_id]
    assert_task_start(state, task_id)
    if not {"static", "review"} <= set(task["requires"]):
        raise ValueError("Full-run source tasks need static and review obligations")
    issues = store.acceptance.validate_task(state, task_id, require_evidence=False)
    if issues:
        raise ValueError("Source mapping incomplete: " + "; ".join(issues))
    missing = set(store.missing_evidence(state, task_id)) & LOCAL_KINDS
    if missing:
        raise ValueError("Missing local source checks: " + ", ".join(sorted(missing)))
    if (
        PHASES.index(task["phase"]) < PHASES.index(state["workflow"]["phase"])
        and not note.strip()
    ):
        raise ValueError(
            "Late dependency repair needs a note explaining the discovered dependency"
        )
    if task["phase"] == "patch":
        task["final_patch_review"] = state["workflow"]["phase"] in {
            "patch",
            "validation",
        } and any(
            e["kind"] == "review"
            and e.get("workflow_phase") in {"patch", "validation"}
            and not e.get("stale")
            and e["result"] == "passed"
            for e in task["evidence"]
        )
    task["source_signature"] = store.source_signature(state, task_id)


def advance(state, note):
    if state["inputs"]["mode"] != "full":
        raise ValueError("Phase advancement applies to full runs")
    current = state["workflow"]["phase"]
    if current == "validation":
        raise ValueError("Already in final validation; use finish after acceptance")
    pending = phase_issues(state)
    analysis = _store().planning.analysis_issues(state)
    if analysis:
        raise ValueError("; ".join(analysis))
    if pending:
        raise ValueError("Source adaptation incomplete: " + ", ".join(pending))
    if not note.strip():
        raise ValueError("Phase handoff needs a review note")
    next_phase = PHASES[PHASES.index(current) + 1]
    state["workflow"]["history"].append(
        {"at": _store().now(), "from": current, "to": next_phase, "note": note}
    )
    state["workflow"]["phase"] = next_phase


def assert_check(state, task_id, kind, scope, command=()):
    model_command = any(
        "tools.batched_test.launch" in token
        or "tools/batched_test/launch.py" in token
        or "vllm.entrypoints" in token
        or token == "--load-format"
        or token.startswith("--load-format=")
        for token in command
    )
    model_command = model_command or any(
        Path(command[index]).name == "vllm" and command[index + 1] == "serve"
        for index in range(max(0, len(command) - 1))
    )
    if scope == "component" and model_command:
        raise ValueError("A model launcher cannot be declared as component execution")
    if kind in MODEL_KINDS and scope != "model":
        raise ValueError("Model evidence kinds require --execution-scope model")
    if scope == "model" and kind not in MODEL_KINDS | {"closure"}:
        raise ValueError(
            "Model execution cannot be recorded as an isolated/static check"
        )
    if state["inputs"]["mode"] != "full":
        return
    assert_task_start(state, task_id)
    if scope == "model":
        if state["workflow"]["phase"] != "validation":
            raise ValueError(
                "All complete-model checks, including dummy smoke, wait for final validation"
            )
        pending = phase_issues(state, "patch")
        if pending:
            raise ValueError(
                "All module and patch source adaptation must finish first: "
                + ", ".join(pending)
            )


def put_unverified(state, document):
    rows = document.get("unverified")
    if not isinstance(rows, list):
        raise ValueError("Provide an unverified list")
    for row in rows:
        if not isinstance(row, dict) or any(
            not isinstance(row.get(key), str) or not row[key].strip()
            for key in ("feature", "chip", "reason", "resume_when")
        ):
            raise ValueError(
                "Unverified coverage needs feature, chip, reason and resume_when"
            )
        if (
            not isinstance(row.get("files"), list)
            or not row["files"]
            or not all(isinstance(path, str) for path in row["files"])
            or not set(row["files"]) <= state["inventory"].keys()
        ):
            raise ValueError("Unverified coverage must identify inventoried files")
        if row.get("status") != "unverified":
            raise ValueError("Unavailable hardware coverage must remain unverified")
    state["workflow"]["unverified"] = rows
    state["completed_at"] = None


def replan(state, task_id, files, phase, owner, note):
    if not note.strip():
        raise ValueError("Task replan needs a review note")
    store = _store()
    task = state["tasks"][task_id]
    previous = {
        key: task.get(key) for key in ("files", "phase", "owner", "upstream_mapping")
    }
    store.invalidate(state, {task_id}, "Task ownership/files replanned")
    task["files"] = sorted({store.relative_file(state, path) for path in files})
    task.pop("upstream_mapping", None)
    task["needs_replan"] = False
    task["status"] = "todo"
    register_task(state, task_id, phase, owner)
    task["notes"].append(
        {"at": store.now(), "text": note, "previous_assignment": previous}
    )


def finish_issues(state):
    if state["inputs"]["mode"] != "full":
        return []
    if "workflow" not in state:
        return ["Missing ordered workflow; migrate the run"]
    synchronize(state)
    errors = _store().planning.analysis_issues(
        state
    ) + _store().model_coverage.finish_issues(state)
    if state["workflow"]["phase"] != "validation":
        errors.append("Full runs must reach final validation")
    pending = [
        path for path, row in state["inventory"].items() if row["status"] != "done"
    ]
    if pending:
        errors.append("Files are not adapted and verified: " + ", ".join(pending))
    errors.extend(
        "Source adaptation is not current: " + path
        for path in phase_issues(state, "patch")
    )
    return errors
