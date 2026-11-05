# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Recorded acceptance contracts; this module never imports candidate packages.

The gates verify declarations, provenance and measured results. They cannot establish
that an agent chose a sufficient support matrix or an independent reference.
"""

import ast
import copy
import difflib
import hashlib
import json
import math
from pathlib import Path
import regex as re
import subprocess


PRODUCTION_KINDS = {
    "closure",
    "weight_loading",
    "e2e",
    "batched_test",
    "distributed",
    "graph",
}
REPORT_KINDS = PRODUCTION_KINDS | {"component_numeric"}
DISPOSITIONS = {"retain", "update", "migrate", "remove", "unsupported", "platform_only"}


def _store():
    if __package__:
        from . import state
    else:
        import state
    return state


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _json(value):
    # Round-trip also detaches caller-owned objects. NaN/Infinity are not JSON facts.
    return json.loads(json.dumps(value, allow_nan=False))


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(label + " must be a nonempty string")
    return value


def _relative(value, label):
    value = _text(value, label)
    path = Path(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or ".git" in path.parts
        or str(path) == "."
    ):
        raise ValueError(
            label + " must be a relative source path without Git internals"
        )
    return str(path)


def _inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Source path escapes its declared checkout: " + relative)
    return path


def _rows(state, task_id=None):
    rows = state.get("support_matrix", {}).values()
    return [row for row in rows if task_id is None or task_id in row["tasks"]]


def _matches(expected, actual):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _matches(value, actual[key])
            for key, value in expected.items()
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(expected) == len(actual)
            and all(_matches(a, b) for a, b in zip(expected, actual))
        )
    return type(expected) is type(actual) and expected == actual


def put_matrix(state, document):
    """Replace the declared matrix after validating it, without invalidating evidence."""
    rows = document.get("rows") if isinstance(document, dict) else document
    if not isinstance(rows, list) or not rows:
        raise ValueError("Support matrix needs a nonempty rows list")
    result = {}
    configurations = {}
    for raw in rows:
        if not isinstance(raw, dict):
            raise ValueError("Each support row must be an object")
        row = _json(raw)
        key = _text(row.get("id"), "Support row id")
        if key in result:
            raise ValueError("Duplicate support row: " + key)
        if row.get("status") not in {"supported", "unsupported"}:
            raise ValueError("Support row status must be supported or unsupported")
        if not isinstance(row.get("configuration"), dict) or not row["configuration"]:
            raise ValueError("Support rows need a nonempty structured configuration")
        if (
            not isinstance(row.get("tasks"), list)
            or not row["tasks"]
            or not all(
                isinstance(task, str) and task in state["tasks"]
                for task in row["tasks"]
            )
        ):
            raise ValueError("Support rows must name existing acceptance tasks")
        if (
            not isinstance(row.get("requires"), list)
            or not row["requires"]
            or not all(
                isinstance(kind, str) and kind in _store().KINDS
                for kind in row["requires"]
            )
        ):
            raise ValueError("Support rows need declared supported evidence kinds")
        if row["status"] == "unsupported":
            _text(row.get("rejection_reason"), "Unsupported row rejection_reason")
            if not set(row["requires"]) & (PRODUCTION_KINDS | {"contract"}):
                raise ValueError(
                    "Unsupported rows require a contract or production rejection check"
                )
        if "dispatch" in row and not row["dispatch"]:
            raise ValueError("Expected dispatch cannot be empty")
        expectation = {"configuration": row["configuration"], "status": row["status"]}
        if "dispatch" in row:
            expectation["dispatch"] = row["dispatch"]
        for task in row["tasks"]:
            if task in configurations and configurations[task] != expectation:
                raise ValueError(
                    "One task cannot certify conflicting support configurations: "
                    + task
                )
            configurations[task] = expectation
        row["tasks"] = sorted(set(row["tasks"]))
        row["requires"] = sorted(set(row["requires"]))
        result[key] = row
    state["support_matrix"] = result


def _git(root, *args):
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError("Cannot inspect upstream source: " + str(exc)) from exc
    if result.returncode:
        raise ValueError(
            "Cannot verify upstream reference: "
            + result.stderr.decode(errors="replace").strip()
        )
    return result.stdout


def _has_git(root):
    # An exported tree may live inside an unrelated parent repository. Only borrow
    # Git metadata when this selected source root is itself a checkout/worktree.
    return (root / ".git").exists()


def _symbol_present(content, path, symbol):
    if symbol in {"<module>", "*"}:
        return True
    if path.endswith(".py"):
        try:
            tree = ast.parse(content.decode("utf-8"))
        except (SyntaxError, UnicodeError) as exc:
            raise ValueError(
                "Cannot inspect upstream Python symbol: " + str(exc)
            ) from exc
        names = set()

        def walk(node, prefix=""):
            for child in ast.iter_child_nodes(node):
                name = getattr(child, "name", None)
                if isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    qualified = prefix + name
                    names.add(qualified)
                    walk(child, qualified + ".")
                else:
                    if isinstance(child, (ast.Assign, ast.AnnAssign)):
                        targets = (
                            child.targets
                            if isinstance(child, ast.Assign)
                            else [child.target]
                        )
                        for assigned in targets:
                            names.update(
                                prefix + sub.id
                                for sub in ast.walk(assigned)
                                if isinstance(sub, ast.Name)
                            )
                    walk(child, prefix)

        walk(tree)
        module = path.removesuffix(".py").replace("/", ".")
        if symbol.startswith(module + "."):
            symbol = symbol[len(module) + 1 :]
        return symbol in names
    # For native code this checks a declared symbol spelling, not its semantics.
    return (
        re.search(
            r"\b" + re.escape(symbol.split(".")[-1]) + r"\b",
            content.decode(errors="replace"),
        )
        is not None
    )


def _reference(state, reference, *, target):
    if not isinstance(reference, dict):
        raise ValueError("Upstream and previous references must be objects")
    ref = _json(reference)
    selected = Path(state["inputs"]["vllm_source"]).resolve()
    if ref.get("absent") is True:
        _text(ref.get("reason"), "Absent upstream reference reason")
        if "path" in ref:
            _relative(ref["path"], "Absent upstream path")
        if target and _has_git(selected):
            revision = (
                _git(selected, "rev-parse", "--verify", "HEAD^{commit}")
                .decode()
                .strip()
            )
            if ref.get("resolved_revision") not in {None, revision}:
                raise ValueError(
                    "Absent-counterpart target revision moved; review and replace the mapping"
                )
            ref.update(source=str(selected), resolved_revision=revision)
        return ref, None
    revision = _text(ref.get("revision"), "Upstream revision")
    if revision.startswith("-") or "\x00" in revision:
        raise ValueError("Invalid upstream revision")
    path = _relative(ref.get("path"), "Upstream path")
    symbol = _text(ref.get("symbol"), "Upstream symbol")
    root = Path(ref.get("source", selected)).resolve()
    if target and root != selected:
        raise ValueError("Target upstream source must match the selected vLLM checkout")
    if _has_git(root):
        resolved = (
            _git(root, "rev-parse", "--verify", revision + "^{commit}").decode().strip()
        )
        if ref.get("resolved_revision") not in {None, resolved}:
            raise ValueError(
                "Recorded upstream revision moved; review and replace the mapping"
            )
        if target:
            target_ref = state["inputs"].get("target_revision", "HEAD")
            target_revision = (
                _git(root, "rev-parse", "--verify", target_ref + "^{commit}")
                .decode()
                .strip()
            )
            if resolved != target_revision:
                raise ValueError(
                    "Upstream mapping revision differs from the selected target revision"
                )
        content = _git(root, "show", resolved + ":" + path)
        ref["resolved_revision"] = resolved
        if target:
            working = _inside(root, path)
            if not working.is_file() or working.read_bytes() != content:
                raise ValueError(
                    "Target working-tree source differs from the mapped upstream revision"
                )
    else:
        content = _inside(root, path).read_bytes()
        ref["resolved_revision"] = revision
    actual_hash = hashlib.sha256(content).hexdigest()
    if target and not re.fullmatch(r"[0-9a-f]{64}", str(ref.get("sha256", ""))):
        raise ValueError("Target upstream references need the exact source sha256")
    if ref.get("sha256") is not None and ref["sha256"] != actual_hash:
        raise ValueError("Upstream source sha256 does not match: " + path)
    if not _symbol_present(content, path, symbol):
        raise ValueError(
            "Upstream symbol was not found in the mapped source: " + symbol
        )
    ref["sha256"] = actual_hash
    ref["path"] = path
    ref["source"] = str(root)
    return ref, content


def _mapping_records(state, task_id, document):
    records = document.get("mappings") if isinstance(document, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError("Task upstream mapping needs a nonempty mappings list")
    result = []
    seen = set()
    task_files = set(state["tasks"][task_id]["files"])
    local_root = Path(state["inputs"]["metax_source"]).resolve()
    for raw in records:
        if not isinstance(raw, dict):
            raise ValueError("Each upstream mapping must be an object")
        item = _json(raw)
        local = _relative(item.get("local"), "Local mapping path")
        if local not in task_files:
            raise ValueError("Mapped local file is not assigned to this task: " + local)
        identity = (local, item.get("local_symbol"))
        if identity in seen:
            raise ValueError("Duplicate local file/symbol mapping: " + local)
        if item.get("local_symbol") is not None:
            _text(item["local_symbol"], "Local mapping symbol")
        seen.add(identity)
        item["local"] = local
        if item.get("disposition") not in DISPOSITIONS:
            raise ValueError("Mapping needs an explicit adaptation disposition")
        for field in ("metax_semantics", "reason", "removal_condition"):
            _text(item.get(field), "Mapping " + field)
        local_path = _inside(local_root, local)
        if not local_path.is_file() and item["disposition"] != "remove":
            raise ValueError(
                "Missing local files require a remove disposition: " + local
            )
        for name, target in (("upstream", True), ("previous", False)):
            try:
                item[name], _ = _reference(state, item.get(name), target=target)
            except OSError as exc:
                raise ValueError("Cannot read mapped source: " + str(exc)) from exc
        result.append(item)
    missing = task_files - {local for local, symbol in seen}
    if missing:
        raise ValueError(
            "Task files lack upstream mappings: " + ", ".join(sorted(missing))
        )
    return {"mappings": result}


def put_mapping(state, task_id, document):
    state["tasks"][task_id]["upstream_mapping"] = _mapping_records(
        state, task_id, document
    )
    if state.get("run_dir"):
        refresh_artifacts(Path(state["run_dir"]), state)


def _number(value, label):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(label + " must be a finite nonnegative number")
    return value


def put_numeric(state, task_id, document):
    contract = _json(document)
    if not isinstance(contract, dict):
        raise ValueError("Numerical contract must be an object")
    reference = contract.get("reference")
    if not isinstance(reference, dict):
        raise ValueError(
            "Numerical contract needs an independent reference declaration"
        )
    _text(reference.get("description"), "Reference description")
    _text(reference.get("revision"), "Reference revision")
    if type(contract.get("seed")) is not int:
        raise ValueError("Numerical seed must be an integer")
    tolerances = contract.get("tolerances")
    if not isinstance(tolerances, dict):
        raise ValueError("Numerical contract needs atol and rtol")
    for name in ("atol", "rtol"):
        _number(tolerances.get(name), name)
    for policy in ("nan_policy", "inf_policy"):
        if contract.get(policy) not in {"forbid", "match_reference"}:
            raise ValueError(policy + " must be forbid or match_reference")
    cases = contract.get("cases")
    if (
        not isinstance(cases, list)
        or not cases
        or not all(isinstance(case, str) and case.strip() for case in cases)
        or len(set(cases)) != len(cases)
    ):
        raise ValueError("Numerical cases must be unique nonempty case names")
    state["tasks"][task_id]["numerical_contract"] = contract


def task_identity(state, task_id):
    task = state["tasks"][task_id]
    return {
        "model_samples": {
            row["model_sample"]: state.get("model_inventory", {})
            .get(row["model_sample"], {})
            .get("config_hash")
            for row in _rows(state, task_id)
            if row.get("model_sample")
        },
        "workflow_assignment": {
            key: task.get(key) for key in ("phase", "owner", "exception_note")
        },
        "support_matrix": sorted(_rows(state, task_id), key=lambda row: row["id"]),
        "upstream_mapping": task.get("upstream_mapping"),
        "numerical_contract": task.get("numerical_contract"),
    }


def _numerical_errors(contract, report):
    result = []
    numerical = report.get("numerical")
    if not isinstance(numerical, dict):
        return ["Check must report measured numerical results for the frozen contract"]
    if numerical.get("contract_digest") != _digest(contract):
        result.append(
            "Numerical contract digest differs from the recorded acceptance contract"
        )
    cases = numerical.get("cases")
    if not isinstance(cases, list) or not all(isinstance(case, dict) for case in cases):
        return result + ["Numerical report needs a cases list"]
    observed = [case.get("id") for case in cases]
    if len(set(str(case) for case in observed)) != len(observed) or set(
        str(case) for case in observed
    ) != set(contract["cases"]):
        result.append("Numerical report must cover every declared case exactly once")
    for case in cases:
        label = str(case.get("id", "unnamed"))
        if case.get("passed") is not True:
            result.append("Numerical case did not pass: " + label)
        try:
            error = _number(case.get("max_abs"), label + " max_abs")
            scale = _number(case.get("reference_scale"), label + " reference_scale")
            threshold = (
                contract["tolerances"]["atol"] + contract["tolerances"]["rtol"] * scale
            )
            if not math.isfinite(threshold) or error > threshold:
                result.append("Numerical error exceeds frozen tolerance: " + label)
        except ValueError as exc:
            result.append(str(exc))
        for value, policy in (("nan", "nan_policy"), ("inf", "inf_policy")):
            count = case.get(value + "_count")
            if type(count) is not int or count < 0:
                result.append(
                    "Numerical case needs a nonnegative " + value + "_count: " + label
                )
            elif contract[policy] == "forbid" and count:
                result.append("Numerical " + value + " policy failed: " + label)
            elif (
                contract[policy] == "match_reference"
                and case.get(value + "_matches_reference") is not True
            ):
                result.append(
                    "Numerical "
                    + value
                    + " pattern/sign comparison is missing: "
                    + label
                )
    return result


def validate_report(state, task_id, kind, report):
    """Validate actual child observations, never a coordinator-supplied label."""
    rows = _rows(state, task_id)
    required = kind in REPORT_KINDS or bool(rows and kind == "contract")
    if not required:
        return []
    if not isinstance(report, dict):
        return ["Check must write a structured actual configuration report"]
    errors = []
    if kind == "batched_test":
        batch = report.get("batched_test", {})
        mode = (
            batch.get("validation_mode", "accuracy")
            if isinstance(batch, dict)
            else None
        )
        configuration = report.get("configuration")
        expected_mode = (
            configuration.get("validation_mode", "accuracy")
            if isinstance(configuration, dict)
            else None
        )
        if mode != expected_mode or mode not in {"accuracy", "dummy_smoke"}:
            errors.append(
                "Batched validation mode differs from the declared model acceptance"
            )
        if mode == "dummy_smoke" and batch.get("semantic_accuracy_claim") is not False:
            errors.append("Dummy smoke must explicitly disclaim semantic accuracy")
        cases = batch.get("cases") if isinstance(batch, dict) else None
        if not isinstance(cases, list) or not cases:
            errors.append("Batched inference needs nonempty measured cases")
        else:
            ids = [case.get("id") for case in cases if isinstance(case, dict)]
            if (
                len(ids) != len(cases)
                or any(not isinstance(key, str) or not key for key in ids)
                or len(set(ids)) != len(ids)
            ):
                errors.append("Batched cases need unique nonempty IDs")
            if any(
                not isinstance(case, dict)
                or case.get("status") != "passed"
                or type(case.get("correct_ratio")) not in {int, float}
                or case["correct_ratio"] != 1.0
                for case in cases
            ):
                errors.append(
                    "Every batched inference case must pass with correct_ratio 1.0"
                )
        artifacts = batch.get("artifacts", []) if isinstance(batch, dict) else []
        names = []
        if not isinstance(artifacts, list):
            errors.append("Batched artifacts must be a list")
            artifacts = []
        for artifact in artifacts:
            if (
                not isinstance(artifact, dict)
                or not isinstance(artifact.get("path"), str)
                or not Path(artifact["path"]).is_absolute()
            ):
                errors.append(
                    "Batched artifacts need absolute paths and content hashes"
                )
                continue
            path = Path(artifact["path"])
            names.append(path.name)
            if (
                not artifact.get("sha256")
                or _store().file_hash(path) != artifact["sha256"]
            ):
                errors.append(
                    "Batched output artifact is missing or changed: " + str(path)
                )
            if path.name.endswith(".manifest.json") and path.is_file():
                try:
                    manifest = json.loads(path.read_text())
                    identities = manifest.get("validation_identities", {})
                    if expected_mode == "dummy_smoke" and (
                        not batch.get("model_tag")
                        or manifest.get("validation_modes", {}).get(batch["model_tag"])
                        != "dummy_smoke"
                    ):
                        errors.append(
                            "Dummy smoke needs the matching model tag and validation mode in its manifest"
                        )
                    expected = _store().validation_identity(state)
                    if (
                        manifest.get("schema_version") != 2
                        or not identities
                        or any(
                            identity.get("source") != expected["source"]
                            or identity.get("environment") != expected["environment"]
                            or not identity.get("checkpoint")
                            or not identity.get("inputs")
                            for identity in identities.values()
                        )
                    ):
                        errors.append(
                            "Batched manifest does not bind the current source/environment/checkpoint/inputs"
                        )
                except (ValueError, AttributeError, TypeError):
                    errors.append("Invalid batched validation identity manifest")
        if "result.json" not in names or not any(
            name.endswith(".manifest.json") for name in names
        ):
            errors.append(
                "Batched acceptance needs result.json and a fingerprint manifest"
            )
    if kind in REPORT_KINDS and not rows:
        errors.append(
            "Production/numerical checks need a structured support-matrix assignment"
        )
    observed = report.get("configuration")
    if not isinstance(observed, dict) or not observed:
        errors.append("Check must report the actual observed configuration")
    for row in rows:
        expected = copy.deepcopy(row["configuration"])
        actual = copy.deepcopy(observed) if isinstance(observed, dict) else {}
        if "dispatch" in expected:
            if "dispatch" not in actual:
                actual["dispatch"] = report.get("dispatch")
            if not _matches(expected["dispatch"], report.get("dispatch")):
                errors.append("Actual dispatch differs from support row: " + row["id"])
        if not _matches(expected, actual):
            errors.append("Actual configuration differs from support row: " + row["id"])
        if "dispatch" in row and not _matches(row["dispatch"], report.get("dispatch")):
            errors.append("Actual dispatch differs from support row: " + row["id"])
        if row["status"] == "unsupported":
            if (
                kind not in PRODUCTION_KINDS | {"contract"}
                or report.get("rejected") is not True
            ):
                errors.append(
                    "Unsupported row needs an asserted rejection result: " + row["id"]
                )
            if (
                not isinstance(report.get("rejection_reason"), str)
                or not report["rejection_reason"].strip()
            ):
                errors.append(
                    "Rejection check must record the actual rejection reason: "
                    + row["id"]
                )
        elif report.get("rejected") is True:
            errors.append("Supported configuration was rejected: " + row["id"])
    contract = state["tasks"][task_id].get("numerical_contract")
    if kind == "component_numeric" and not contract:
        errors.append(
            "Component numerical checks require a recorded numerical contract"
        )
    if contract and kind in REPORT_KINDS:
        errors.extend(_numerical_errors(contract, report))
    return errors


def _current_evidence(state, task_id):
    task = state["tasks"][task_id]
    latest = {}
    for evidence in task["evidence"]:
        if not evidence.get("stale"):
            latest[(evidence["kind"], evidence.get("execution_scope"))] = evidence
    return [
        evidence
        for evidence in latest.values()
        if evidence.get("result") == "passed"
        and evidence.get("signature")
        == _store().check_signature(
            state, task_id, evidence["kind"], evidence.get("execution_scope")
        )
        and evidence.get("environment") == state["environment"]["fingerprint"]
        and _store().evidence_inputs_current(evidence)
        and not validate_report(
            state,
            task_id,
            evidence["kind"],
            evidence.get("report", evidence.get("origins")),
        )
    ]


def validate_task(state, task_id, *, require_evidence=True):
    task = state["tasks"][task_id]
    errors = _store().planning.source_issues(
        state, task_id
    ) + _store().model_coverage.review_issues(state, task_id)
    if task["files"]:
        try:
            _mapping_records(state, task_id, task.get("upstream_mapping"))
        except (ValueError, OSError) as exc:
            errors.append(str(exc))
    if "component_numeric" in task["requires"] and not task.get("numerical_contract"):
        errors.append("Component numerical tasks need a frozen numerical contract")
    if require_evidence:
        current = {evidence["kind"] for evidence in _current_evidence(state, task_id)}
        for kind in sorted(_store().effective_requirements(state, task_id)):
            if kind not in current:
                errors.append("Missing current valid acceptance evidence: " + kind)
    return errors


def validate_finish(state):
    errors = _store().workflow.finish_issues(state)
    for task_id in state["tasks"]:
        errors.extend(task_id + ": " + error for error in validate_task(state, task_id))
    rows = _rows(state)
    if state["inputs"]["mode"] == "full" and not rows:
        errors.append(
            "Full upgrades need a declared support matrix with production e2e obligations"
        )
    if state["inputs"]["mode"] == "full" and not any(
        row["status"] == "supported" for row in rows
    ):
        errors.append(
            "Full upgrades need at least one supported row with production e2e evidence"
        )
    for row in rows:
        observations = [
            evidence
            for task_id in row["tasks"]
            for evidence in _current_evidence(state, task_id)
        ]
        kinds = {evidence["kind"] for evidence in observations}
        required = set(row["requires"])
        if row["status"] == "supported" and state["inputs"]["mode"] == "full":
            required.update({"e2e", "batched_test"})
        if row.get("model_sample") and _store().model_coverage.valid_skip(
            state, row["model_sample"]
        ):
            required -= _store().model_coverage.MODEL_KINDS
        for kind in sorted(required - kinds):
            errors.append(
                row["id"] + ": missing current " + kind + " acceptance evidence"
            )
        if row["status"] == "unsupported" and not any(
            evidence["kind"] in PRODUCTION_KINDS | {"contract"}
            and evidence.get("report", evidence.get("origins", {})).get("rejected")
            is True
            for evidence in observations
        ):
            errors.append(row["id"] + ": missing current asserted rejection evidence")
    return errors


def refresh_artifacts(run_dir, state):
    """Write reviewable target-vs-local diffs; existing evidence remains historical."""
    run_dir = Path(run_dir)
    root = Path(state["inputs"]["metax_source"])
    store = _store()
    for task_id, task in state["tasks"].items():
        mapping = task.get("upstream_mapping")
        if not mapping:
            continue
        observed = {}
        upstream_root = Path(state["inputs"]["vllm_source"])
        for item in mapping["mappings"]:
            reference = item["upstream"]
            for label, source, relative in (
                ("local", root, item["local"]),
                ("upstream", upstream_root, reference.get("path")),
            ):
                if relative is not None:
                    try:
                        observed[label + ":" + relative] = store.file_hash(
                            _inside(source, relative)
                        )
                    except (ValueError, OSError) as exc:
                        observed[label + ":" + relative] = str(exc)
        if _has_git(upstream_root):
            try:
                observed["upstream_revision"] = (
                    _git(upstream_root, "rev-parse", "HEAD").decode().strip()
                )
            except ValueError as exc:
                observed["upstream_revision"] = str(exc)
        identity = _digest(
            {"task": task_id, "mapping": mapping, "observed_sources": observed}
        )
        folder = run_dir / "review" / identity
        folder.mkdir(parents=True, exist_ok=True)
        records = []
        for index, item in enumerate(mapping["mappings"]):
            record = copy.deepcopy(item)
            try:
                _, content = _reference(state, item["upstream"], target=True)
                local = _inside(root, item["local"])
                local_content = local.read_bytes() if local.is_file() else b""
                upstream = content or b""
                diff = "".join(
                    difflib.unified_diff(
                        upstream.decode(errors="replace").splitlines(keepends=True),
                        local_content.decode(errors="replace").splitlines(
                            keepends=True
                        ),
                        fromfile="upstream/" + item["upstream"].get("path", "<absent>"),
                        tofile="metax/" + item["local"],
                    )
                )
                name = f"{index:04d}.diff"
                store.atomic_write(folder / name, diff)
                record.update(
                    diff=name, local_sha256=hashlib.sha256(local_content).hexdigest()
                )
            except (ValueError, OSError) as exc:
                record["error"] = str(exc)
            records.append(record)
        store.write_json(
            folder / "mapping.json", {"task": task_id, "mappings": records}
        )
        task["review_artifact"] = str((folder / "mapping.json").relative_to(run_dir))
