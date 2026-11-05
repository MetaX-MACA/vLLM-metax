#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Environment gate, durable TODO and reproducible checks for MetaX upgrades."""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid

if __package__:
    from . import (
        acceptance,
        environment,
        lifecycle,
        processes,
        preparation_gate,
        workflow,
        state as store,
    )
else:
    import acceptance
    import environment
    import lifecycle
    import processes
    import preparation_gate
    import state as store
    import workflow


PRODUCTION_KINDS = {
    "closure",
    "weight_loading",
    "e2e",
    "batched_test",
    "distributed",
    "graph",
}


def check_policy(run_dir, state, allow_pending=False):
    if state["schema_version"] != store.SCHEMA_VERSION:
        raise ValueError("Legacy run: use migrate --note TEXT before continuing")
    current = lifecycle.runtime_identity()
    if state.get("policy_identity") != current:
        state["policy_change"] = {
            "previous": state.get("policy_identity"),
            "current": current,
            "detected_at": store.now(),
        }
        store.invalidate(state, set(state["tasks"]), "Tool or acceptance rules changed")
        store.save(run_dir, state)
    if state.get("policy_change") and not allow_pending:
        raise ValueError(
            "Tool or acceptance rules changed; review and run accept-policy-change --note TEXT"
        )


def finish_initialization(run_dir, state):
    phase = state.get("initialization", {}).get("phase", "ready")
    if phase == "baseline":
        lifecycle.prepare_baseline(run_dir, state)
        phase = state["initialization"]["phase"]
    if phase == "environment":
        snapshot(run_dir, state, collect(state))
        state["initialization"]["phase"] = "ready"
        store.save(run_dir, state)


def migrate(run_dir, state, note):
    version = state["schema_version"]
    if version not in {1, 2, 3}:
        raise ValueError("Only schema 1, 2 or 3 runs need this migration")
    if not note.strip():
        raise ValueError("Migration needs a review note")
    original = (Path(run_dir) / "state.json").read_text()
    backup = (
        Path(run_dir) / "migrations" / f"schema-{version}-{store.digest(state)}.json"
    )
    store.atomic_write(backup, original)
    state.update(
        schema_version=store.SCHEMA_VERSION,
        initialization=state.get("initialization", {"phase": "ready"}),
        support_matrix=state.get("support_matrix", {}),
        policy_identity=lifecycle.runtime_identity(),
        policy_history=[{"at": store.now(), "note": note, "backup": str(backup)}],
    )
    workflow.initialize(state)
    baseline = Path(run_dir) / "baseline/source-manifest.json"
    if baseline.is_file():
        for path, content_hash in json.loads(baseline.read_text()).items():
            if content_hash is not None:
                row = state["inventory"].setdefault(
                    path,
                    {"phase": workflow.classify(path), "initial_hash": content_hash},
                )
                row["initial_hash"] = content_hash
    for task in state["tasks"].values():
        if len({workflow.classify(path) for path in task["files"]}) > 1:
            task["needs_replan"] = True
    for task_id in state["tasks"]:
        try:
            if state["tasks"][task_id].get("needs_replan"):
                raise ValueError("Review and split legacy files across source phases")
            workflow.register_task(state, task_id)
        except ValueError as exc:
            task = state["tasks"][task_id]
            task["phase"] = max(
                (workflow.classify(path) for path in task["files"]),
                key=workflow.PHASES.index,
                default="validation",
            )
            task["owner"] = "unassigned"
            task["needs_replan"] = True
            task["notes"].append(
                {"at": store.now(), "text": "Migration needs task-replan: " + str(exc)}
            )
    store.invalidate(
        state, set(state["tasks"]), "Migrated acceptance rules; revalidate"
    )
    store.save(run_dir, state)


def collect(state, full=True):
    inputs = state["inputs"]
    with preparation_gate.kernel_environment(state.get("run_dir")):
        return environment.collect_environment(
            Path(inputs["vllm_source"]),
            Path(inputs["metax_source"]),
            Path(inputs["runtime_cwd"]),
            include_collect_env=full,
        )


def snapshot(run_dir, state, report):
    fingerprint = environment.fingerprint(report)
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    name = f"environment/{store.digest(report)}.json"
    target = Path(run_dir) / name
    if not target.exists():
        store.atomic_write(target, encoded)
    previous = state.get("environment", {})
    changed = previous.get("fingerprint") != fingerprint
    approval = previous.get("confirmation") if not changed else None
    if preparation_gate.uses_plan(run_dir, state):
        try:
            approval = preparation_gate.confirmation(run_dir, state, fingerprint)
        except ValueError:
            approval = None
    state["environment"] = {
        "fingerprint": fingerprint,
        "snapshot": name,
        "snapshot_sha256": store.file_hash(target),
        "confirmation": approval,
        "checked_at": store.now(),
        "issues": report["readiness"]["issues"],
        "full_collection": report.get("details", {}).get("collection_profile")
        != "identity",
    }
    if changed and previous:
        store.invalidate(state, set(state["tasks"]), "Environment identity changed")
    # The full JSON is authoritative; this small view helps user review the dump.
    summary = {
        "fingerprint": fingerprint,
        "snapshot": name,
        "identity": report["identity"],
        "readiness": report["readiness"],
        "note": "Full upgrades require an approved environment-check plan and successful automatic verification.",
    }
    store.write_json(Path(run_dir) / "ENVIRONMENT.json", summary)
    store.atomic_write(
        Path(run_dir) / "ENVIRONMENT.txt", environment.format_environment(report) + "\n"
    )
    return fingerprint


def approved_snapshot(run_dir, state):
    env = state["environment"]
    path = Path(run_dir) / env["snapshot"]
    if store.file_hash(path) != env["snapshot_sha256"]:
        raise ValueError(
            "Environment snapshot changed or is missing; refresh and verify preparation"
            if preparation_gate.uses_plan(run_dir, state)
            else "Environment snapshot changed or is missing; refresh and reconfirm"
        )
    confirmation = env.get("confirmation")
    if not confirmation or confirmation["fingerprint"] != env["fingerprint"]:
        raise ValueError(
            "Environment is not user-confirmed; review the dump and record the user's explicit reply"
        )


def assert_ready(run_dir, state):
    check_policy(run_dir, state)
    if state.get("initialization", {}).get("phase", "ready") != "ready":
        raise ValueError("Initialization is incomplete; run resume or refresh")
    plan_confirmation = preparation_gate.confirmation(
        run_dir, state, state["environment"]["fingerprint"]
    )
    if plan_confirmation is not None:
        state["environment"]["confirmation"] = plan_confirmation
    approved_snapshot(run_dir, state)
    report = collect(state, full=False)
    if environment.fingerprint(report) != state["environment"]["fingerprint"]:
        snapshot(run_dir, state, report)
        store.save(run_dir, state)
        raise ValueError(
            "Environment drift detected; refresh and verify against the approved preparation plan"
            if preparation_gate.uses_plan(run_dir, state)
            else "Environment drift detected; collect full diagnostics with refresh and obtain confirmation"
        )
    state["environment"]["checked_at"] = store.now()


def origins_valid(state, kind, path, task_id=None):
    needs_report = kind in PRODUCTION_KINDS | {"component_numeric"}
    if not needs_report and not path.is_file():
        return True, None
    if not path.is_file():
        return (
            False,
            "Check must write actual runtime origins to METAX_UPGRADE_ORIGINS_FILE",
        )
    try:
        origins = json.loads(path.read_text())
        if not isinstance(origins, dict):
            raise ValueError("Expected a JSON object")
        if kind in PRODUCTION_KINDS:
            if preparation_gate.kernel_variables(state.get("run_dir")):
                _, plan = preparation_gate.preparation.load_plan(state["run_dir"])
                issue = preparation_gate.preparation.backend_issue(plan, origins)
                if issue:
                    return False, issue
            for package, source in (
                ("vllm", "vllm_source"),
                ("vllm_metax", "metax_source"),
            ):
                root = Path(state["inputs"][source]) / package
                origin = Path(origins[package]).resolve()
                if not origin.is_file() or not origin.is_relative_to(root.resolve()):
                    return (
                        False,
                        f"{package} origin does not match the confirmed checkout",
                    )
            if not origins.get("dispatch"):
                return (
                    False,
                    "Production check must record and assert the selected dispatch",
                )
        if kind == "component_numeric" and not origins.get("callables"):
            return (
                False,
                "Component numerical check must record actual callable/extension origins",
            )
        if task_id is not None:
            issues = acceptance.validate_report(state, task_id, kind, origins)
            if issues:
                return False, "; ".join(issues)
        return True, origins
    except (KeyError, TypeError, ValueError, OSError) as exc:
        return False, "Invalid origins: " + str(exc)


def run_check(run_dir, state, args):
    assert_ready(run_dir, state)
    store.synchronize(state)
    if not args.command:
        raise ValueError("Provide a command after --")
    command = args.command[1:] if args.command[0] == "--" else args.command
    if not command or args.timeout <= 0:
        raise ValueError("Provide a command and a positive timeout")
    task = state["tasks"][args.id]
    scope = getattr(args, "execution_scope", None)
    if scope is None:
        if state["inputs"]["mode"] == "full":
            raise ValueError(
                "Full-run checks need --execution-scope component or model"
            )
        scope = "model" if args.kind in workflow.MODEL_KINDS else "component"
    workflow.assert_check(state, args.id, args.kind, scope, command)
    issues = acceptance.validate_task(state, args.id, require_evidence=False)
    if issues:
        raise ValueError("Task acceptance incomplete: " + "; ".join(issues))
    if args.kind in PRODUCTION_KINDS | {"component_numeric"} and not any(
        args.id in row["tasks"] for row in state.get("support_matrix", {}).values()
    ):
        raise ValueError(
            "Production/numerical checks need a structured support-matrix assignment"
        )
    if args.kind == "component_numeric" and not task.get("numerical_contract"):
        raise ValueError("Component numerical checks need a frozen numerical contract")
    if args.kind in PRODUCTION_KINDS:
        incomplete = [
            key
            for key in task["depends_on"]
            if state["tasks"][key]["status"] not in {"implemented", "done"}
        ]
        if incomplete:
            raise ValueError(
                "Production closure has unimplemented dependencies: "
                + ", ".join(incomplete)
            )
    if args.configuration != task["configuration"]:
        raise ValueError(
            "Check configuration must equal the task configuration; split configuration cases into tasks"
        )
    signature = store.check_signature(state, args.id, args.kind, scope)
    runtime_cwd = Path(state["inputs"]["runtime_cwd"])
    input_files = [
        str(p if p.is_absolute() else runtime_cwd / p)
        for p in getattr(args, "input_file", [])
    ]
    input_hashes = lifecycle.evidence_inputs(
        command, state["inputs"]["runtime_cwd"], input_files
    )
    evidence_id = uuid.uuid4().hex
    folder = Path(run_dir) / "checks" / evidence_id
    folder.mkdir(parents=True)
    store.write_json(
        folder / "acceptance.json",
        {
            **acceptance.task_identity(state, args.id),
            "policy_identity": state["policy_identity"],
        },
    )
    origins = folder / "origins.json"
    environment_vars = dict(os.environ)
    environment_vars.update(preparation_gate.kernel_variables(run_dir))
    validation_identity = store.validation_identity(state)
    identity_file = folder / "validation-identity.json"
    store.write_json(identity_file, validation_identity)
    environment_vars["METAX_UPGRADE_VALIDATION_IDENTITY_FILE"] = str(
        identity_file.resolve()
    )
    environment_vars["METAX_UPGRADE_ORIGINS_FILE"] = str(origins.resolve())
    environment_vars["METAX_UPGRADE_CHECK_ID"] = evidence_id
    numerical = task.get("numerical_contract")
    if numerical:
        contract_file = folder / "numerical-contract.json"
        store.write_json(contract_file, numerical)
        environment_vars["METAX_UPGRADE_NUMERICAL_CONTRACT_FILE"] = str(
            contract_file.resolve()
        )
        environment_vars["METAX_UPGRADE_NUMERICAL_CONTRACT_DIGEST"] = store.digest(
            numerical
        )
    if args.kind not in PRODUCTION_KINDS:
        environment_vars["VLLM_PLUGINS"] = ""
    evidence = {
        "id": evidence_id,
        "kind": args.kind,
        "execution_scope": scope,
        "workflow_phase": state.get("workflow", {}).get("phase"),
        "result": "interrupted",
        "at": store.now(),
        "command": command,
        "cwd": state["inputs"]["runtime_cwd"],
        "configuration": args.configuration,
        "signature": signature,
        "validation_identity": validation_identity,
        "environment": state["environment"]["fingerprint"],
        "policy": state["policy_identity"]["fingerprint"],
        "artifact": str(folder.relative_to(run_dir)),
        "input_files": input_files,
        "input_hashes": input_hashes,
        "stale": False,
        "isolation": "normal_plugin_configuration"
        if args.kind in PRODUCTION_KINDS
        else "plugins_disabled",
    }
    task["evidence"].append(evidence)
    # Persist before starting: a hard interruption must leave a visible pending check.
    store.save(run_dir, state)
    timed_out = False
    try:
        with (
            (folder / "stdout.log").open("w") as stdout,
            (folder / "stderr.log").open("w") as stderr,
        ):
            worker = Path(__file__).with_name("check_worker.py")
            worker_command = [
                sys.executable,
                str(worker),
                json.dumps(processes.identity(os.getpid())),
                str((folder / "release").resolve()),
                str(args.timeout),
                *command,
            ]
            process = subprocess.Popen(
                worker_command,
                cwd=evidence["cwd"],
                env=environment_vars,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            evidence["process"] = processes.identity(process.pid)
            # Persist identity before release; derived TODO work must not consume
            # the supervisor's bounded startup handshake window.
            store.save(run_dir, state, render_todo=False)
            # A killed coordinator cannot launch an unrecorded GPU workload.
            (folder / "release").touch()
            try:
                returncode = process.wait(timeout=args.timeout + 1)
            except (subprocess.TimeoutExpired, KeyboardInterrupt):
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait()
                timed_out = True
    except OSError as exc:
        returncode = 127
        (folder / "stderr.log").write_text(str(exc))
    worker_report = folder / "worker-result.json"
    worker_result = (
        json.loads(worker_report.read_text()) if worker_report.is_file() else None
    )
    timed_out = timed_out or bool(
        worker_result and worker_result["reason"] == "timeout"
    )
    valid_origins, origin_detail = origins_valid(state, args.kind, origins, args.id)
    changed = store.synchronize(state)
    # A check that rewrites its subject cannot certify the post-edit source.
    unchanged = (
        not changed
        and signature == store.check_signature(state, args.id, args.kind, scope)
        and store.evidence_inputs_current(evidence)
        and lifecycle.runtime_identity() == state.get("policy_identity")
    )
    post_report = collect(state, full=False)
    same_environment = environment.fingerprint(post_report) == evidence["environment"]
    if not same_environment:
        snapshot(run_dir, state, post_report)
    evidence.update(
        {
            "result": "passed"
            if returncode == 0 and valid_origins and unchanged and same_environment
            else "failed",
            "returncode": returncode,
            "timed_out": timed_out,
            "worker": worker_result,
            "origins": origin_detail,
            "report": origin_detail if isinstance(origin_detail, dict) else None,
            "source_unchanged": unchanged,
            "environment_unchanged": same_environment,
            "finished_at": store.now(),
        }
    )
    if evidence["result"] == "failed" and task["status"] == "done":
        store.invalidate(
            state, {args.id}, "Previously completed task failed a new check"
        )
    store.write_json(folder / "result.json", evidence)
    store.save(run_dir, state)
    print(json.dumps(evidence, indent=2))
    return 0 if evidence["result"] == "passed" else 1


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="action", required=True)
    for name in (
        "init",
        "refresh",
        "confirm",
        "assert-ready",
        "task-add",
        "task-set",
        "scope-set",
        "contract-put",
        "run-check",
        "checkpoint",
        "resume",
        "status",
        "finish",
        "matrix-put",
        "mapping-put",
        "numeric-put",
        "migrate",
        "accept-policy-change",
        "phase-next",
        "coverage-put",
        "task-replan",
        "analysis-put",
        "checks-put",
        "model-review",
        "model-bind",
        "model-skip",
    ):
        command = commands.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
        if name == "init":
            for option in ("vllm-source", "metax-source", "runtime-cwd"):
                command.add_argument("--" + option, type=Path, required=True)
            command.add_argument("--scope", required=True)
            command.add_argument(
                "--mode", choices=("full", "component"), default="full"
            )
        elif name == "confirm":
            command.add_argument("--fingerprint", required=True)
            command.add_argument("--user-response-file", type=Path, required=True)
            command.add_argument("--acknowledge-issues", action="store_true")
        elif name == "task-add":
            command.add_argument("--id", required=True)
            command.add_argument("--title", required=True)
            command.add_argument("--files", action="append", default=[])
            command.add_argument("--depends-on", action="append", default=[])
            command.add_argument(
                "--requires",
                choices=sorted(store.KINDS),
                action="append",
                required=True,
            )
            command.add_argument("--configuration", required=True)
            command.add_argument("--next-action", required=True)
            command.add_argument("--phase", choices=workflow.PHASES)
            command.add_argument("--owner")
            command.add_argument("--exception-note", default="")
        elif name == "task-set":
            command.add_argument("--id", required=True)
            command.add_argument(
                "--status", choices=sorted(store.STATUSES), required=True
            )
            command.add_argument("--note", default="")
            command.add_argument("--next-action", default="")
        elif name == "task-replan":
            command.add_argument("--id", required=True)
            command.add_argument("--files", action="append", required=True)
            command.add_argument("--phase", choices=workflow.PHASES, required=True)
            command.add_argument("--owner", required=True)
            command.add_argument("--note", required=True)
        elif name == "scope-set":
            command.add_argument("--scope", required=True)
        elif name in {"contract-put", "matrix-put", "mapping-put", "numeric-put"}:
            command.add_argument("--file", type=Path, required=True)
            if name in {"mapping-put", "numeric-put"}:
                command.add_argument("--id", required=True)
            if name == "numeric-put":
                command.add_argument("--note", default="")
        elif name in {"migrate", "accept-policy-change"}:
            command.add_argument("--note", required=True)
        elif name == "phase-next":
            command.add_argument("--note", required=True)
        elif name == "coverage-put":
            command.add_argument("--file", type=Path, required=True)
        elif name in {"analysis-put", "model-bind", "checks-put"}:
            command.add_argument("--file", type=Path, required=True)
            if name == "checks-put":
                command.add_argument("--id", required=True)
        elif name == "model-review":
            command.add_argument("--id", required=True)
            command.add_argument("--entry", action="append", required=True)
            command.add_argument("--spec", action="append", required=True)
        elif name == "model-skip":
            command.add_argument("--id", required=True)
            command.add_argument(
                "--reason-code",
                choices=("insufficient_gpus", "insufficient_vram"),
                required=True,
            )
            command.add_argument("--required", type=int, required=True)
            command.add_argument("--available", type=int, required=True)
            command.add_argument("--evidence-file", type=Path, required=True)
            command.add_argument("--note", required=True)
        elif name == "run-check":
            command.add_argument("--id", required=True)
            command.add_argument("--kind", choices=sorted(store.KINDS), required=True)
            command.add_argument("--configuration", required=True)
            command.add_argument("--timeout", type=int, default=120)
            command.add_argument("--input-file", type=Path, action="append", default=[])
            command.add_argument("--execution-scope", choices=("component", "model"))
            command.add_argument("command", nargs=argparse.REMAINDER)
        elif name == "checkpoint":
            command.add_argument("--note", required=True)
            command.add_argument("--next-action", required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    args.run_dir = args.run_dir.resolve()
    with store.locked(args.run_dir):
        if args.action == "init":
            if (args.run_dir / "state.json").exists():
                raise ValueError(
                    "Run already exists; use resume, never overwrite its progress"
                )
            inputs = {
                key: str(getattr(args, key).resolve())
                for key in ("vllm_source", "metax_source", "runtime_cwd")
            }
            if not all(Path(p).is_dir() for p in inputs.values()):
                raise ValueError("Source and runtime directories must exist")
            inputs["scope"] = args.scope
            inputs["mode"] = args.mode
            inputs["environment_protocol"] = "preparation-plan-v1"
            state = {
                "schema_version": store.SCHEMA_VERSION,
                "revision": 0,
                "created_at": store.now(),
                "run_dir": str(args.run_dir),
                "initialization": {"phase": "baseline"},
                "policy_identity": lifecycle.runtime_identity(),
                "policy_history": [],
                "support_matrix": {},
                "inputs": inputs,
                "tasks": {},
                "contracts": {},
                "checkpoints": [],
                "unassigned_changes": [],
                "completed_at": None,
                "environment": {
                    "fingerprint": "not-collected",
                    "snapshot": "environment/pending.json",
                    "snapshot_sha256": None,
                    "confirmation": None,
                    "issues": ["Environment collection has not finished"],
                    "full_collection": False,
                },
            }
            manifest = args.run_dir / "baseline/source-manifest.json"
            state["source_manifest"] = (
                json.loads(manifest.read_text())
                if manifest.is_file()
                else store.source_manifest(state)
            )
            workflow.initialize(state)
            # Persist intent before any baseline archive or environment work.
            store.save(args.run_dir, state)
            finish_initialization(args.run_dir, state)
        else:
            state = store.load(args.run_dir)
            if args.action == "migrate":
                migrate(args.run_dir, state, args.note)
                print(store.render(state))
                return 0
            if (
                state["schema_version"] != store.SCHEMA_VERSION
                and args.action == "status"
            ):
                print(store.render(state))
                print("Legacy run: migrate --note TEXT before continuing")
                return 0
            check_policy(
                args.run_dir,
                state,
                allow_pending=args.action
                in {
                    "status",
                    "resume",
                    "refresh",
                    "accept-policy-change",
                    "checkpoint",
                    "task-add",
                },
            )
            initializing = (
                state.get("initialization", {}).get("phase", "ready") != "ready"
            )
            if initializing and args.action in {"resume", "refresh"}:
                finish_initialization(args.run_dir, state)
            elif initializing and args.action not in {"status", "accept-policy-change"}:
                raise ValueError("Initialization is incomplete; run resume or refresh")
            if initializing and args.action == "status":
                print(store.render(state))
                return 0
            changed = store.synchronize(state, recover=args.action == "resume")
            if args.action == "resume":
                for task in state["tasks"].values():
                    for evidence in task["evidence"]:
                        if evidence["result"] == "interrupted":
                            evidence["recovery_cleanup"] = processes.terminate(
                                evidence.get("process")
                            )
                            evidence["stale"] = True
                            evidence["stale_reason"] = (
                                "Interrupted check; rerun after recovery"
                            )
                store.save(args.run_dir, state)
            else:
                # Input-only invalidations also survive a subsequently rejected action.
                store.save(args.run_dir, state)
            if changed:
                acceptance.refresh_artifacts(args.run_dir, state)
                store.save(args.run_dir, state)
            if args.action == "refresh":
                if not initializing:
                    snapshot(args.run_dir, state, collect(state))
            elif args.action == "accept-policy-change":
                if not args.note.strip():
                    raise ValueError("Policy review needs a nonempty note")
                state["policy_history"].append(
                    {
                        "at": store.now(),
                        "note": args.note,
                        "change": state.get("policy_change"),
                    }
                )
                state["policy_identity"] = lifecycle.runtime_identity()
                state.pop("policy_change", None)
            elif args.action == "confirm":
                if preparation_gate.uses_plan(args.run_dir, state):
                    raise ValueError(
                        "Use environment-check: approve the preparation plan once, then verify installation automatically"
                    )
                if not state["environment"].get("full_collection", False):
                    raise ValueError(
                        "Run refresh to collect full diagnostics before asking for confirmation"
                    )
                if args.fingerprint != state["environment"]["fingerprint"]:
                    raise ValueError(
                        "Fingerprint does not match the snapshot presented for confirmation"
                    )
                report = collect(state, full=False)
                if environment.fingerprint(report) != args.fingerprint:
                    snapshot(args.run_dir, state, report)
                    store.save(args.run_dir, state)
                    raise ValueError(
                        "Environment changed since collection; refresh and ask for a new confirmation"
                    )
                reply = args.user_response_file.read_text().strip()
                if not reply:
                    raise ValueError("The actual explicit user reply is required")
                if state["environment"]["issues"] and not args.acknowledge_issues:
                    raise ValueError(
                        "Snapshot has issues; the user must acknowledge the disclosed gaps before confirmation"
                    )
                response = {
                    "at": store.now(),
                    "fingerprint": args.fingerprint,
                    "reply": reply,
                    "acknowledged_issues": bool(args.acknowledge_issues),
                }
                response_name = (
                    "environment/approval-" + store.digest(response) + ".json"
                )
                store.write_json(args.run_dir / response_name, response)
                state["environment"]["confirmation"] = {
                    **response,
                    "record": response_name,
                }
            elif args.action in {"assert-ready", "resume"}:
                try:
                    assert_ready(args.run_dir, state)
                finally:
                    store.save(args.run_dir, state)
            elif args.action == "task-add":
                store.add_task(
                    state,
                    args.id,
                    args.title,
                    args.files,
                    args.depends_on,
                    args.requires,
                    args.configuration,
                    args.next_action,
                )
                workflow.register_task(
                    state, args.id, args.phase, args.owner, args.exception_note
                )
            elif args.action == "task-set":
                if args.status in {"in_progress", "implemented", "done"}:
                    assert_ready(args.run_dir, state)
                    workflow.assert_task_start(state, args.id)
                if args.status == "implemented":
                    workflow.mark_implemented(state, args.id, args.note)
                if args.status == "done" and state["inputs"]["mode"] == "full":
                    if state["tasks"][args.id]["files"] and not workflow.source_ready(
                        state, args.id
                    ):
                        raise ValueError(
                            "Mark source implemented with current local evidence before final completion"
                        )
                store.set_task(state, args.id, args.status, args.note, args.next_action)
            elif args.action == "phase-next":
                assert_ready(args.run_dir, state)
                workflow.advance(state, args.note)
            elif args.action == "task-replan":
                assert_ready(args.run_dir, state)
                workflow.replan(
                    state, args.id, args.files, args.phase, args.owner, args.note
                )
            elif args.action == "coverage-put":
                assert_ready(args.run_dir, state)
                workflow.put_unverified(state, json.loads(args.file.read_text()))
            elif args.action == "analysis-put":
                assert_ready(args.run_dir, state)
                store.planning.put_analysis(state, json.loads(args.file.read_text()))
            elif args.action == "checks-put":
                assert_ready(args.run_dir, state)
                store.planning.put_checks(
                    state, args.id, json.loads(args.file.read_text())
                )
            elif args.action == "model-review":
                assert_ready(args.run_dir, state)
                workflow.assert_task_start(state, args.id)
                store.model_coverage.review_model(
                    args.run_dir, state, args.id, args.entry, args.spec
                )
            elif args.action == "model-bind":
                assert_ready(args.run_dir, state)
                store.model_coverage.bind(state, json.loads(args.file.read_text()))
            elif args.action == "model-skip":
                assert_ready(args.run_dir, state)
                store.model_coverage.skip(
                    state,
                    args.id,
                    args.reason_code,
                    args.required,
                    args.available,
                    args.evidence_file,
                    args.note,
                )
            elif args.action == "scope-set":
                assert_ready(args.run_dir, state)
                store.set_scope(state, args.scope)
            elif args.action == "contract-put":
                assert_ready(args.run_dir, state)
                store.put_contract(state, json.loads(args.file.read_text()))
            elif args.action in {"matrix-put", "mapping-put", "numeric-put"}:
                assert_ready(args.run_dir, state)
                document = json.loads(args.file.read_text())
                previous = {
                    key: store.task_signature(state, key) for key in state["tasks"]
                }
                if args.action == "matrix-put":
                    acceptance.put_matrix(state, document)
                elif args.action == "mapping-put":
                    acceptance.put_mapping(state, args.id, document)
                else:
                    prior_numeric = state["tasks"][args.id].get("numerical_contract")
                    if (
                        prior_numeric
                        and prior_numeric != document
                        and not args.note.strip()
                    ):
                        raise ValueError(
                            "Changing numerical acceptance needs --note with a technical reason"
                        )
                    acceptance.put_numeric(state, args.id, document)
                    if prior_numeric != document:
                        state.setdefault("acceptance_history", []).append(
                            {
                                "at": store.now(),
                                "task": args.id,
                                "kind": "numerical_contract",
                                "note": args.note,
                                "previous": prior_numeric,
                                "current": document,
                            }
                        )
                affected = {
                    key
                    for key, signature in previous.items()
                    if signature != store.task_signature(state, key)
                }
                if affected:
                    store.invalidate(
                        state,
                        affected,
                        "Acceptance definition updated: " + args.action,
                        kinds=workflow.MODEL_KINDS
                        if args.action == "matrix-put"
                        else None,
                    )
            elif args.action == "run-check":
                return run_check(args.run_dir, state, args)
            elif args.action == "checkpoint":
                state["checkpoints"].append(
                    {
                        "at": store.now(),
                        "note": args.note,
                        "next_action": args.next_action,
                    }
                )
            elif args.action == "finish":
                assert_ready(args.run_dir, state)
                issues = acceptance.validate_finish(state)
                if issues:
                    raise ValueError("Acceptance incomplete: " + "; ".join(issues))
                if not state["tasks"] or state["unassigned_changes"]:
                    raise ValueError(
                        "Need a nonempty covered worklist with no unassigned changes"
                    )
                if any(
                    task["status"] != "done" or store.missing_evidence(state, key)
                    for key, task in state["tasks"].items()
                ):
                    raise ValueError(
                        "Every task needs done status and current required evidence"
                    )
                covered = {p for task in state["tasks"].values() for p in task["files"]}
                if (
                    state["inputs"]["mode"] == "full"
                    and set(state["inventory"]) - covered
                ):
                    raise ValueError(
                        "Production/build/requirements inventory is not fully assigned to disposition tasks"
                    )
                state["completed_at"] = store.now()
        store.save(args.run_dir, state)
        print(store.render(state))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
