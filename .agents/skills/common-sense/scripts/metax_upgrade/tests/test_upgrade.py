# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Offline gate/recovery checks; never load the target plugin or GPU stack."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from metax_upgrade import acceptance, environment, state as store, upgrade, workflow


def synthetic_mapping(run_dir, task_id):
    """Explicit platform-only dispositions for synthetic offline fixture files."""
    state = store.load(run_dir)
    source = Path(state["inputs"]["metax_source"])
    acceptance.put_mapping(
        state,
        task_id,
        {
            "mappings": [
                {
                    "local": path,
                    "disposition": "platform_only"
                    if (source / path).is_file()
                    else "remove",
                    "upstream": {
                        "absent": True,
                        "reason": "Synthetic fixture has no upstream implementation",
                    },
                    "previous": {
                        "absent": True,
                        "reason": "Synthetic fixture has no previous upstream release",
                    },
                    "metax_semantics": "Exercise the offline state and validation machinery",
                    "reason": "Test fixture, independent of production adaptation semantics",
                    "removal_condition": "Remove when this isolated test fixture is no longer needed",
                }
                for path in state["tasks"][task_id]["files"]
            ]
        },
    )
    task = state["tasks"][task_id]
    report_kinds = set(task["requires"]) & acceptance.REPORT_KINDS
    if report_kinds:
        rows = [
            row
            for key, row in state.get("support_matrix", {}).items()
            if key != task_id
        ]
        rows.append(
            {
                "id": task_id,
                "status": "supported",
                "configuration": {"fixture": task["configuration"]},
                "tasks": [task_id],
                "requires": sorted(report_kinds),
            }
        )
        acceptance.put_matrix(state, {"rows": rows})
    store.save(run_dir, state)


def prepare_full_validation(cli, run):
    """Give synthetic acceptance scenarios real local checks and phase handoffs."""
    state = store.load(run)
    for task_id, task in state["tasks"].items():
        task["requires"] = sorted(set(task["requires"]) | {"static", "review"})
        workflow.register_task(state, task_id)
    workflow.synchronize(state)
    missing = [path for path, row in state["inventory"].items() if not row["tasks"]]
    if missing:
        store.add_task(
            state,
            "remaining-files",
            "Review remaining synthetic files",
            missing,
            [],
            ["static", "review"],
            "fixture",
            "Review",
        )
        workflow.register_task(state, "remaining-files")
    store.save(run, state)
    if missing:
        synthetic_mapping(run, "remaining-files")
    current = store.load(run)
    plan = Path(run) / "analysis.json"
    plan.write_text(
        json.dumps(
            {
                "files": list(current["inventory"]),
                "models": [],
                "edges": [],
                "cycles": [],
                "summary": "Synthetic independent file ownership",
            }
        )
    )
    cli("analysis-put", "--file", plan)
    for task_id in current["tasks"]:
        checks = Path(run) / (task_id + "-checks.json")
        checks.write_text(
            json.dumps(
                {
                    "changes": ["none"],
                    "reason": "Synthetic static fixture with no numerical operations",
                }
            )
        )
        cli("checks-put", "--id", task_id, "--file", checks)
    while store.load(run)["workflow"]["phase"] != "validation":
        state = store.load(run)
        phase = state["workflow"]["phase"]
        for task_id, task in state["tasks"].items():
            if task["phase"] != phase:
                continue
            for kind in ("static", "review"):
                cli(
                    "run-check",
                    "--id",
                    task_id,
                    "--kind",
                    kind,
                    "--execution-scope",
                    "component",
                    "--configuration",
                    task["configuration"],
                    "--",
                    sys.executable,
                    "-c",
                    "pass",
                )
            cli("task-set", "--id", task_id, "--status", "implemented")
        cli("phase-next", "--note", "Synthetic source phase complete")


class UpgradeTests(unittest.TestCase):
    def setUp(self):
        # Synthetic phase fixtures have no real source installations; the actual
        # preparation gate is exercised separately in environment-check tests.
        prep_gate = patch.object(
            upgrade.preparation_gate, "confirmation", return_value=None
        )
        prep_gate.start()
        self.addCleanup(prep_gate.stop)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.metax = self.root / "metax"
        self.vllm = self.root / "upstream"
        self.run = self.root / "run"
        (self.metax / "vllm_metax").mkdir(parents=True)
        (self.vllm / "vllm").mkdir(parents=True)
        for path in (
            self.metax / "vllm_metax/__init__.py",
            self.vllm / "vllm/__init__.py",
        ):
            path.write_text("# clean\n")
        self.file = self.metax / "vllm_metax/op.py"
        self.file.write_text("value = 1\n")
        self.report = {
            "identity": {
                "sdk": {"sdk_versions": ["3"]},
                "python": {"executable": sys.executable},
            },
            "readiness": {"issues": []},
        }
        self.collector = patch.object(
            environment,
            "collect_environment",
            side_effect=lambda *a, **k: json.loads(json.dumps(self.report)),
        )
        self.collector.start()
        self.addCleanup(self.collector.stop)
        self.cli(
            "init",
            "--vllm-source",
            self.vllm,
            "--metax-source",
            self.metax,
            "--runtime-cwd",
            self.metax,
            "--scope",
            "offline test",
            "--mode",
            "component",
        )

    def cli(self, action, *args):
        argv = [action, "--run-dir", str(self.run), *(str(arg) for arg in args)]
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                return upgrade.main(argv)
            finally:
                if action == "init" and (self.run / "state.json").exists():
                    legacy = store.load(self.run)
                    legacy["inputs"].pop("environment_protocol", None)
                    store.save(self.run, legacy)

    def approve(self):
        reply = self.root / "reply.txt"
        reply.write_text("I confirm the displayed environment and the disclosed gaps.")
        self.cli(
            "confirm",
            "--fingerprint",
            environment.fingerprint(self.report),
            "--user-response-file",
            reply,
            "--acknowledge-issues",
        )

    def task(
        self, key="op", dependency=None, requires="static", files="vllm_metax/op.py"
    ):
        args = [
            "--id",
            key,
            "--title",
            key,
            "--files",
            files,
            "--requires",
            requires,
            "--configuration",
            "BF16 single GPU",
            "--next-action",
            "Compare target contracts",
        ]
        if dependency:
            args.extend(["--depends-on", dependency])
        self.cli("task-add", *args)
        synthetic_mapping(self.run, key)

    def check(self, code="print('ok')", key="op", kind="static"):
        return self.cli(
            "run-check",
            "--id",
            key,
            "--kind",
            kind,
            "--configuration",
            "BF16 single GPU",
            "--",
            sys.executable,
            "-c",
            code,
        )

    def test_state_only_save_persists_without_rendering(self):
        state = store.load(self.run)
        revision = state["revision"]
        todo = (self.run / "TODO.md").read_bytes()
        with patch.object(store, "render", side_effect=AssertionError("rendered")):
            store.save(self.run, state, render_todo=False)
        self.assertEqual(store.load(self.run)["revision"], revision + 1)
        self.assertEqual((self.run / "TODO.md").read_bytes(), todo)

    def test_check_registers_identity_before_release_without_rendering(self):
        self.approve()
        self.task()
        original_render = store.render

        def guarded_render(state):
            for task in state["tasks"].values():
                for evidence in task["evidence"]:
                    if evidence.get("process") and evidence["result"] == "interrupted":
                        self.assertTrue(
                            (self.run / evidence["artifact"] / "release").is_file()
                        )
            return original_render(state)

        code = (
            "import json; from pathlib import Path; "
            f"state=json.loads(Path({str(self.run / 'state.json')!r}).read_text()); "
            "assert state['tasks']['op']['evidence'][-1]['process']['pid'] > 0"
        )
        with patch.object(store, "render", side_effect=guarded_render):
            self.check(code)
        self.assertEqual(
            store.load(self.run)["tasks"]["op"]["evidence"][-1]["result"], "passed"
        )

    def test_todo_skips_input_rehash_but_acceptance_checks_it(self):
        self.approve()
        self.task()
        self.check()
        state = store.load(self.run)
        with patch.object(
            store, "evidence_inputs_current", return_value=False
        ) as current:
            store.render(state)
            current.assert_not_called()
            self.assertEqual(store.missing_evidence(state, "op"), ["static"])
            current.assert_called_once()

    def test_gate_prevents_work_and_validation_without_confirmation(self):
        self.task()
        for action, args in (
            ("task-set", ("--id", "op", "--status", "in_progress")),
            (
                "run-check",
                (
                    "--id",
                    "op",
                    "--kind",
                    "static",
                    "--configuration",
                    "BF16 single GPU",
                    "--",
                    sys.executable,
                    "-c",
                    "pass",
                ),
            ),
        ):
            with self.assertRaisesRegex(ValueError, "not user-confirmed"):
                self.cli(action, *args)
        self.assertEqual(store.load(self.run)["tasks"]["op"]["status"], "todo")

    def test_initial_dirty_source_is_preserved(self):
        with zipfile.ZipFile(self.run / "baseline/metax-source.zip") as archive:
            self.assertEqual(archive.read("vllm_metax/op.py"), b"value = 1\n")
        self.file.write_text("value = 2\n")
        with zipfile.ZipFile(self.run / "baseline/metax-source.zip") as archive:
            self.assertEqual(archive.read("vllm_metax/op.py"), b"value = 1\n")

    def test_evidence_latest_failure_wins_and_completion_needs_evidence(self):
        self.approve()
        self.task()
        with self.assertRaisesRegex(ValueError, "Missing current"):
            self.cli("task-set", "--id", "op", "--status", "done")
        self.assertEqual(self.check(), 0)
        self.cli("task-set", "--id", "op", "--status", "done")
        self.assertEqual(self.check("raise RuntimeError('regression')"), 1)
        with self.assertRaisesRegex(ValueError, "Missing current"):
            self.cli("task-set", "--id", "op", "--status", "done")

    def test_changed_producer_invalidates_consumer(self):
        self.approve()
        other = self.metax / "vllm_metax/consumer.py"
        other.write_text("pass\n")
        self.task()
        self.task("consumer", dependency="op", files="vllm_metax/consumer.py")
        for key in ("op", "consumer"):
            self.assertEqual(self.check(key=key), 0)
            self.cli("task-set", "--id", key, "--status", "done")
        self.file.write_text("value = 2\n")
        self.cli("resume")
        state = store.load(self.run)
        self.assertEqual(state["tasks"]["consumer"]["status"], "implemented")
        self.assertTrue(state["tasks"]["consumer"]["evidence"][0]["stale"])

    def test_resume_recovers_interrupted_work_and_repairs_todo(self):
        self.approve()
        self.task()
        self.cli(
            "task-set",
            "--id",
            "op",
            "--status",
            "in_progress",
            "--next-action",
            "Inspect partial diff",
        )
        (self.run / "TODO.md").write_text("stale projection")
        self.cli("resume")
        self.assertEqual(store.load(self.run)["tasks"]["op"]["status"], "todo")
        self.assertIn("Inspect partial diff", (self.run / "TODO.md").read_text())

    def test_environment_drift_removes_approval_and_preserves_progress(self):
        self.approve()
        self.task()
        self.assertEqual(self.check(), 0)
        self.cli("task-set", "--id", "op", "--status", "done")
        self.report["identity"]["sdk"]["sdk_versions"] = ["4"]
        with self.assertRaisesRegex(ValueError, "drift"):
            self.cli("assert-ready")
        state = store.load(self.run)
        self.assertIsNone(state["environment"]["confirmation"])
        self.assertEqual(state["tasks"]["op"]["status"], "implemented")

    def test_production_check_rejects_absent_and_wrong_wheel_origins(self):
        self.approve()
        self.task(requires="e2e")
        self.assertEqual(self.check(kind="e2e"), 1)
        payload = {
            "vllm": str(self.vllm / "vllm/__init__.py"),
            "vllm_metax": "/wrong/site-packages/vllm_metax/__init__.py",
            "dispatch": {"op": "real"},
        }
        code = (
            "import json,os; open(os.environ['METAX_UPGRADE_ORIGINS_FILE'],'w').write("
            + repr(json.dumps(payload))
            + ")"
        )
        self.assertEqual(self.check(code, kind="e2e"), 1)

    def test_component_finish_does_not_claim_full_upgrade(self):
        self.approve()
        self.task()
        self.assertEqual(self.check(), 0)
        self.cli("task-set", "--id", "op", "--status", "done")
        self.cli("finish")
        self.assertIsNotNone(store.load(self.run)["completed_at"])
        self.assertIn("Mode: component", (self.run / "TODO.md").read_text())

    def test_scope_set_records_change_and_invalidates_evidence(self):
        self.approve()
        self.task()
        self.assertEqual(self.check(), 0)
        self.cli("task-set", "--id", "op", "--status", "done")

        self.cli("scope-set", "--scope", "expanded model matrix")

        state = store.load(self.run)
        self.assertEqual(state["inputs"]["scope"], "expanded model matrix")
        self.assertEqual(state["scope_history"][-1]["previous"], "offline test")
        self.assertEqual(state["tasks"]["op"]["status"], "implemented")
        self.assertTrue(state["tasks"]["op"]["evidence"][-1]["stale"])
        self.assertIn(
            "Scope: expanded model matrix", (self.run / "TODO.md").read_text()
        )

    def test_check_that_changes_subject_cannot_pass(self):
        self.approve()
        self.task()
        self.assertEqual(
            self.check(
                "from pathlib import Path; Path('vllm_metax/op.py').write_text('value = 99\\n')"
            ),
            1,
        )

    def test_spawned_background_work_is_cleaned_and_check_fails(self):
        self.approve()
        self.task()
        code = "import subprocess,sys; subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])"
        self.assertEqual(self.check(code), 1)
        evidence = store.load(self.run)["tasks"]["op"]["evidence"][-1]
        self.assertEqual(evidence["worker"]["reason"], "leftover_processes")

    def test_worker_timeout_is_recorded(self):
        self.approve()
        self.task()
        result = self.cli(
            "run-check",
            "--id",
            "op",
            "--kind",
            "static",
            "--configuration",
            "BF16 single GPU",
            "--timeout",
            "1",
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        )
        self.assertEqual(result, 1)
        self.assertTrue(
            store.load(self.run)["tasks"]["op"]["evidence"][-1]["timed_out"]
        )

    def test_done_consumer_requires_done_dependency(self):
        self.approve()
        self.task()
        self.task("consumer", dependency="op")
        self.assertEqual(self.check(key="consumer"), 0)
        with self.assertRaisesRegex(ValueError, "Dependencies are not done"):
            self.cli("task-set", "--id", "consumer", "--status", "done")

    def test_full_mode_cannot_finish_from_static_only_tasks(self):
        self.approve()
        self.task()
        self.assertEqual(self.check(), 0)
        self.cli("task-set", "--id", "op", "--status", "done")
        state = store.load(self.run)
        state["inputs"]["mode"] = "full"
        state["tasks"]["op"]["files"] = list(state["source_manifest"])
        store.save(self.run, state)
        synthetic_mapping(self.run, "op")
        with self.assertRaisesRegex(ValueError, "production e2e"):
            self.cli("finish")

    def test_interrupted_initial_collection_leaves_refreshable_state(self):
        another = self.root / "interrupted-init"
        with patch.object(
            environment,
            "collect_environment",
            side_effect=TimeoutError("killed collector"),
        ):
            with self.assertRaises(TimeoutError):
                with contextlib.redirect_stdout(io.StringIO()):
                    upgrade.main(
                        [
                            "init",
                            "--run-dir",
                            str(another),
                            "--vllm-source",
                            str(self.vllm),
                            "--metax-source",
                            str(self.metax),
                            "--runtime-cwd",
                            str(self.metax),
                            "--scope",
                            "recovery",
                        ]
                    )
        self.assertFalse(store.load(another)["environment"]["full_collection"])
        with contextlib.redirect_stdout(io.StringIO()):
            upgrade.main(["refresh", "--run-dir", str(another)])
        self.assertEqual(
            store.load(another)["environment"]["fingerprint"],
            environment.fingerprint(self.report),
        )


if __name__ == "__main__":
    unittest.main()
