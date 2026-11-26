# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Exercise ordered source gates and recovery with real files/CLI, without models."""

import json
from pathlib import Path
import sys
import unittest

from metax_upgrade import acceptance, state as store, workflow
from metax_upgrade.tests import test_upgrade as fixtures


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.UpgradeTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.cli = self.fixture.cli
        self.run = self.fixture.run
        self.metax = self.fixture.metax
        self.fixture.approve()
        (self.fixture.vllm / "sample.py").write_text(
            "def encode(items, /, *, options=None):\n    return items, options\n"
        )
        self.files = {
            "platform": ["vllm_metax/platform.py"],
            "registry": ["vllm_metax/registry/config.py"],
            "attention": ["vllm_metax/v1/attention/backend.py"],
            "framework": ["vllm_metax/__init__.py", "vllm_metax/op.py"],
            "models": [
                "vllm_metax/models/example/model.py",
                "vllm_metax/models/example/SPEC.md",
            ],
            "patch": ["vllm_metax/patch/fix.py"],
        }
        for paths in self.files.values():
            for path in paths:
                source = self.metax / path
                source.parent.mkdir(parents=True, exist_ok=True)
                source.write_text("# synthetic source\n")
        state = store.load(self.run)
        state["inputs"]["mode"] = "full"
        state["source_manifest"] = store.source_manifest(state)
        workflow.initialize(state)
        store.save(self.run, state)
        for phase, paths in self.files.items():
            args = [
                "--id",
                phase,
                "--title",
                phase,
                "--phase",
                phase,
                "--requires",
                "static",
                "--requires",
                "review",
                "--configuration",
                "fixture",
                "--next-action",
                "Adapt synthetic source",
            ]
            for path in paths:
                args.extend(["--files", path])
            if phase == "models":
                args.extend(["--requires", "e2e", "--requires", "batched_test"])
            self.cli("task-add", *args)
            fixtures.synthetic_mapping(self.run, phase)
            policy = self.run / (phase + "-checks.json")
            checks = {
                "changes": ["none"],
                "reason": "Synthetic comment-only fixture; no tensor behavior",
            }
            if phase in {"patch", "registry"}:
                checks["no_api_replacements"] = {
                    path: "Comment-only fixture has no exposed callable"
                    for path in paths
                }
            policy.write_text(json.dumps(checks))
            self.cli("checks-put", "--id", phase, "--file", policy)
        current = store.load(self.run)
        plan = self.run / "analysis.json"
        plan.write_text(
            json.dumps(
                {
                    "files": list(current["inventory"]),
                    "models": [],
                    "edges": [],
                    "cycles": [],
                    "summary": "Synthetic independent source files with no inter-module runtime contracts",
                    "upstream_review": fixtures.synthetic_upstream_review(current),
                }
            )
        )
        self.cli("analysis-put", "--file", plan)

    def check(self, task, kind="static", scope="component", code="pass"):
        return self.cli(
            "run-check",
            "--id",
            task,
            "--kind",
            kind,
            "--execution-scope",
            scope,
            "--configuration",
            "fixture",
            "--",
            sys.executable,
            "-c",
            code,
        )

    def implement(self, phase, note="Reviewed complete synthetic source"):
        self.cli("task-set", "--id", phase, "--status", "in_progress")
        if phase == "models":
            self.cli(
                "model-review",
                "--id",
                phase,
                "--entry",
                self.files[phase][0],
                "--spec",
                self.files[phase][1],
            )
        self.assertEqual(self.check(phase), 0)
        self.assertEqual(self.check(phase, "review"), 0)
        self.cli("task-set", "--id", phase, "--status", "implemented", "--note", note)

    def reach(self, target):
        while store.load(self.run)["workflow"]["phase"] != target:
            phase = store.load(self.run)["workflow"]["phase"]
            self.implement(phase)
            self.cli("phase-next", "--note", "All source in this phase reviewed")

    def report_code(self, ratio=1.0):
        report = {
            "vllm": str(self.fixture.vllm / "vllm/__init__.py"),
            "vllm_metax": str(self.metax / "vllm_metax/__init__.py"),
            "dispatch": {"fixture": "local"},
            "configuration": store.load(self.run)["support_matrix"]["models"][
                "configuration"
            ],
        }
        folder = self.run / "batch"
        folder.mkdir(exist_ok=True)
        artifacts = []
        for name in ("result.json", "inference_results.manifest.json"):
            path = folder / name
            payload = {"synthetic_fixture": True}
            if name.endswith(".manifest.json"):
                payload = {
                    "schema_version": 2,
                    "validation_modes": {
                        "fixture": report["configuration"].get(
                            "validation_mode", "accuracy"
                        )
                    },
                    "validation_identities": {
                        "fixture": {
                            **store.validation_identity(store.load(self.run)),
                            "checkpoint": "synthetic",
                            "inputs": "synthetic",
                        }
                    },
                }
            path.write_text(json.dumps(payload))
            artifacts.append({"path": str(path), "sha256": store.file_hash(path)})
        mode = report["configuration"].get("validation_mode", "accuracy")
        report["batched_test"] = {
            "validation_mode": mode,
            "model_tag": "fixture",
            "semantic_accuracy_claim": mode == "accuracy",
            "cases": [{"id": "text", "status": "passed", "correct_ratio": ratio}],
            "artifacts": artifacts,
        }
        return (
            "import json,os; from pathlib import Path; Path(os.environ['METAX_UPGRADE_ORIGINS_FILE']).write_text(json.dumps("
            + repr(report)
            + "))"
        )

    def test_missing_files_block_phase_and_later_model_start(self):
        with self.assertRaisesRegex(ValueError, "Source adaptation incomplete"):
            self.cli("phase-next", "--note", "Cannot skip files")
        with self.assertRaisesRegex(ValueError, "cannot start models"):
            self.cli("task-set", "--id", "models", "--status", "in_progress")
        self.assertIn(
            "vllm_metax/patch/fix.py", json.loads((self.run / "FILES.json").read_text())
        )

    def test_retained_registry_binding_needs_current_api_and_call_evidence(self):
        self.reach("registry")
        local = self.files["registry"][0]
        upstream = self.fixture.vllm / "sample.py"
        source = upstream.read_text()
        (self.metax / local).write_text(source + '\nregistry = {"example": encode}\n')
        state = store.load(self.run)
        mapping = state["tasks"]["registry"]["upstream_mapping"]
        mapping["mappings"][0].update(
            local_symbol="encode",
            disposition="retain",
            upstream={
                "path": "sample.py",
                "symbol": "encode",
                "revision": "HEAD",
                "sha256": store.file_hash(upstream),
            },
        )
        path = self.run / "registry-mapping.json"
        path.write_text(json.dumps(mapping))
        self.cli("mapping-put", "--id", "registry", "--file", path)
        with self.assertRaisesRegex(
            ValueError, "Mapped replacement needs an API contract"
        ):
            self.check("registry")

        policy = self.run / "registry-checks.json"
        policy.write_text(
            json.dumps(
                {
                    "changes": ["none"],
                    "reason": "Verify the retained registry override",
                    "api_contracts": [
                        {
                            "id": "example",
                            "local": local,
                            "local_symbol": "encode",
                            "binding": {
                                "entry": "example",
                                "callable": "fixture.registry.encode",
                            },
                            "mode": "exact",
                            "surface": "python",
                            "target": {"signature": "(items, /, *, options=None)"},
                            "cases": ["valid_call", "invalid_call"],
                        }
                    ],
                }
            )
        )
        self.cli("checks-put", "--id", "registry", "--file", policy)
        self.assertEqual(self.check("registry"), 0)
        self.assertEqual(self.check("registry", "review"), 0)
        self.assertNotEqual(self.check("registry", "contract"), 0)
        with self.assertRaisesRegex(
            ValueError, "Missing local source checks: contract"
        ):
            self.cli("task-set", "--id", "registry", "--status", "implemented")

        probe = f"""
import inspect
import json
import os
from pathlib import Path
import runpy

target = runpy.run_path({str(upstream)!r})['encode']
entry = 'example'
active = runpy.run_path({local!r}, run_name='fixture.registry')['registry'][entry]
valid = active('items', options='explicit') == target('items', options='explicit')
def rejects_keyword(fn):
    try:
        fn(items='items')
    except TypeError:
        return True
    return False
result = {{
    'binding': {{'entry': entry, 'callable': active.__module__ + '.' + active.__qualname__}},
    'upstream': {{'signature': str(inspect.signature(target))}},
    'active': {{'signature': str(inspect.signature(active))}},
    'cases': {{'valid_call': valid, 'invalid_call': rejects_keyword(target) and rejects_keyword(active)}},
}}
Path(os.environ['METAX_UPGRADE_ORIGINS_FILE']).write_text(json.dumps({{'api_contracts': {{entry: result}}}}))
"""
        (self.metax / local).write_text(
            source.replace("options=None", "options=False")
            + '\nregistry = {"example": encode}\n'
        )
        self.assertNotEqual(self.check("registry", "contract", code=probe), 0)
        evidence = store.load(self.run)["tasks"]["registry"]["evidence"][-1]
        self.assertIn("active signature", evidence["origins"])
        (self.metax / local).write_text(source + '\nregistry = {"example": encode}\n')
        self.assertEqual(self.check("registry", "contract", code=probe), 0)
        self.assertEqual(self.check("registry"), 0)
        self.assertEqual(self.check("registry", "review"), 0)
        self.cli("task-set", "--id", "registry", "--status", "implemented")

    def test_model_launchers_and_phase_relabeling_cannot_bypass_source_gates(self):
        with self.assertRaisesRegex(ValueError, "model launcher"):
            self.cli(
                "run-check",
                "--id",
                "platform",
                "--kind",
                "static",
                "--execution-scope",
                "component",
                "--configuration",
                "fixture",
                "--",
                "vllm",
                "serve",
                "dummy-model",
            )
        with self.assertRaisesRegex(ValueError, "Production modules"):
            self.cli(
                "task-add",
                "--id",
                "bypass",
                "--title",
                "Invalid phase",
                "--files",
                "vllm_metax/registry/config.py",
                "--phase",
                "validation",
                "--requires",
                "static",
                "--configuration",
                "fixture",
                "--next-action",
                "Invalid",
            )

        with self.assertRaisesRegex(
            ValueError, "must be adapted by the framework phase"
        ):
            self.cli(
                "task-add",
                "--id",
                "late-framework",
                "--title",
                "Invalid model assignment",
                "--files",
                "vllm_metax/op.py",
                "--phase",
                "models",
                "--requires",
                "static",
                "--configuration",
                "fixture",
                "--next-action",
                "Invalid",
            )

    def test_complete_model_smoke_waits_even_after_model_source_is_ready(self):
        self.reach("models")
        self.implement("models")
        row = store.load(self.run)["inventory"][self.files["models"][0]]
        self.assertTrue(row["source_ready"])
        self.assertEqual(row["status"], "in_progress")
        with self.assertRaisesRegex(ValueError, "wait for final validation"):
            self.check("models", "closure", "model")
        with self.assertRaisesRegex(ValueError, "require --execution-scope model"):
            self.check("models", "e2e", "component")

    def test_late_framework_repair_closes_gate_and_requires_a_reason(self):
        self.reach("models")
        path = self.metax / "vllm_metax/op.py"
        path.write_text("# changed indirect dependency\n")
        self.cli("status")
        with self.assertRaisesRegex(
            ValueError, "Framework source adaptation is incomplete"
        ):
            self.cli("task-set", "--id", "models", "--status", "in_progress")
        self.check("framework")
        self.check("framework", "review")
        with self.assertRaisesRegex(ValueError, "Late dependency repair needs a note"):
            self.cli("task-set", "--id", "framework", "--status", "implemented")
        self.cli(
            "task-set",
            "--id",
            "framework",
            "--status",
            "implemented",
            "--note",
            "Model revealed changed framework dependency",
        )
        self.cli("task-set", "--id", "models", "--status", "in_progress")

    def test_new_and_removed_files_remain_in_inventory(self):
        self.reach("registry")
        old = self.metax / self.files["platform"][0]
        old.unlink()
        added = self.metax / "vllm_metax/new_helper.py"
        added.write_text("# new dependency\n")
        self.cli("resume")
        state = store.load(self.run)
        self.assertFalse(state["inventory"][self.files["platform"][0]]["present"])
        self.assertIn("vllm_metax/new_helper.py", state["inventory"])
        self.assertEqual(state["inventory"]["vllm_metax/new_helper.py"]["owners"], [])
        with self.assertRaisesRegex(ValueError, "dependency analysis"):
            self.cli("phase-next", "--note", "Removed platform needs review")

    def test_early_blocking_patch_still_needs_final_patch_phase_review(self):
        state = store.load(self.run)
        state["tasks"]["patch"]["exception_note"] = (
            "Old patch blocks the platform unit check"
        )
        store.save(self.run, state)
        self.implement("patch")
        self.assertFalse(store.load(self.run)["tasks"]["patch"]["final_patch_review"])
        self.reach("patch")
        with self.assertRaisesRegex(ValueError, "Source adaptation incomplete"):
            self.cli("phase-next", "--note", "Early fix is not the formal review")
        self.implement("patch")
        self.cli("phase-next", "--note", "Removal-first patch review complete")

    def test_batched_accuracy_and_artifact_identity_are_required(self):
        self.reach("validation")
        self.assertEqual(
            self.check("models", "batched_test", "model", self.report_code(0.5)), 1
        )
        # Failure invalidates local evidence; refresh source acceptance first.
        self.implement("models")
        self.assertEqual(
            self.check("models", "batched_test", "model", self.report_code()), 0
        )
        state = store.load(self.run)
        report = state["tasks"]["models"]["evidence"][-1]["report"]
        Path(report["batched_test"]["artifacts"][0]["path"]).write_text("changed")
        self.assertTrue(
            acceptance.validate_report(state, "models", "batched_test", report)
        )

    def test_final_acceptance_and_unverified_chip_coverage_survive_resume(self):
        self.reach("validation")
        path = self.run / "coverage.json"
        path.write_text(
            json.dumps(
                {
                    "unverified": [
                        {
                            "feature": "FP8 cache",
                            "chip": "C5xx",
                            "files": self.files["attention"],
                            "reason": "Chip does not support FP8 execution",
                            "resume_when": "Supported chip available",
                            "status": "unverified",
                        }
                    ]
                }
            )
        )
        self.cli("coverage-put", "--file", path)
        self.assertEqual(self.check("models", "e2e", "model", self.report_code()), 0)
        with self.assertRaisesRegex(ValueError, "batched_test"):
            self.cli("finish")
        self.assertEqual(
            self.check("models", "batched_test", "model", self.report_code()), 0
        )
        for phase in self.files:
            self.cli("task-set", "--id", phase, "--status", "done")
        self.cli("finish")
        self.cli("resume")
        state = store.load(self.run)
        self.assertTrue(state["completed_at"])
        self.assertTrue(
            all(row["status"] == "done" for row in state["inventory"].values())
        )
        self.assertEqual(state["workflow"]["unverified"][0]["feature"], "FP8 cache")
        self.assertIn("UNVERIFIED FP8 cache", (self.run / "TODO.md").read_text())

    def test_schema_two_migration_preserves_baseline_and_stales_evidence(self):
        self.reach("registry")
        state = store.load(self.run)
        state["schema_version"] = 2
        original_archive = (self.run / "baseline/metax-source.zip").read_bytes()
        store.save(self.run, state)
        self.cli("migrate", "--note", "Review legacy run in new source order")
        migrated = store.load(self.run)
        self.assertEqual(migrated["schema_version"], 4)
        self.assertEqual(migrated["workflow"]["phase"], "platform")
        self.assertFalse(workflow.source_ready(migrated, "platform"))
        self.assertTrue(list((self.run / "migrations").glob("schema-2-*")))
        baseline = json.loads((self.run / "baseline/source-manifest.json").read_text())
        self.assertEqual(
            migrated["inventory"]["vllm_metax/op.py"]["initial_hash"],
            baseline["vllm_metax/op.py"],
        )
        self.assertEqual(
            (self.run / "baseline/metax-source.zip").read_bytes(), original_archive
        )

    def test_previous_model_owner_is_normalized_when_resuming_a_run(self):
        state = store.load(self.run)
        self.assertEqual(state["tasks"]["models"]["owner"], "model-adaptation")
        state["tasks"]["models"]["owner"] = "model-upgrade"
        state["policy_identity"] = {"fingerprint": "previous-model-skill"}
        store.write_json(self.run / "state.json", state)

        self.cli("accept-policy-change", "--note", "Reviewed the merged model skill")
        model_file = self.files["models"][0]
        self.cli(
            "task-add",
            "--id",
            "model-config",
            "--title",
            "Review another model configuration",
            "--files",
            model_file,
            "--requires",
            "static",
            "--configuration",
            "another configuration",
            "--next-action",
            "Review the shared model file",
        )
        saved = json.loads((self.run / "state.json").read_text())
        self.assertEqual(saved["tasks"]["models"]["owner"], "model-adaptation")
        self.assertEqual(saved["tasks"]["model-config"]["owner"], "model-adaptation")
        self.assertEqual(saved["inventory"][model_file]["owners"], ["model-adaptation"])

    def test_mixed_legacy_task_can_be_replanned_without_losing_history(self):
        state = store.load(self.run)
        state["schema_version"] = 2
        state["tasks"]["models"]["files"].append("vllm_metax/op.py")
        store.save(self.run, state)
        self.cli("migrate", "--note", "Split cross-phase legacy ownership")
        self.assertTrue(store.load(self.run)["tasks"]["models"]["needs_replan"])
        args = [
            "--id",
            "models",
            "--phase",
            "models",
            "--owner",
            "model-upgrade",
            "--note",
            "Keep only model files; framework remains owned by framework task",
        ]
        for path in self.files["models"]:
            args.extend(["--files", path])
        self.cli("task-replan", *args)
        task = store.load(self.run)["tasks"]["models"]
        self.assertEqual(task["owner"], "model-adaptation")
        self.assertFalse(task["needs_replan"])
        self.assertEqual(task["status"], "todo")
        self.assertIn(
            "vllm_metax/op.py", task["notes"][-1]["previous_assignment"]["files"]
        )
