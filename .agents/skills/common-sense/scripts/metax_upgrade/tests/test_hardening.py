# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Offline CLI scenarios for recovery, rule changes and complete support coverage."""

import contextlib
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from metax_upgrade import environment, lifecycle, state as store, upgrade
from metax_upgrade.tests.test_upgrade import prepare_full_validation, synthetic_mapping


class HardeningTests(unittest.TestCase):
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
        self.vllm = self.root / "vllm-source"
        self.run = self.root / "run"
        for folder in (self.metax / "vllm_metax", self.vllm / "vllm"):
            folder.mkdir(parents=True)
            (folder / "__init__.py").write_text("# synthetic fixture\n")
        for name in ("op", "second"):
            (self.metax / f"vllm_metax/{name}.py").write_text("value = 1\n")
        self.report = {
            "identity": {"sdk": {"sdk_versions": ["synthetic-sdk"]}},
            "readiness": {"issues": []},
        }
        collector = patch.object(
            upgrade, "collect", side_effect=lambda *_a, **_k: copy.deepcopy(self.report)
        )
        collector.start()
        self.addCleanup(collector.stop)
        self.init()
        self.confirm()

    def cli(self, action, *args, run=None):
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                return upgrade.main(
                    [action, "--run-dir", str(run or self.run), *map(str, args)]
                )
            finally:
                legacy_run = run or self.run
                if action == "init" and (legacy_run / "state.json").exists():
                    legacy = store.load(legacy_run)
                    legacy["inputs"].pop("environment_protocol", None)
                    store.save(legacy_run, legacy)

    def init(self, run=None):
        return self.cli(
            "init",
            "--vllm-source",
            self.vllm,
            "--metax-source",
            self.metax,
            "--runtime-cwd",
            self.root,
            "--scope",
            "offline hardening",
            "--mode",
            "component",
            run=run,
        )

    def confirm(self):
        reply = self.root / "actual-user-reply.txt"
        reply.write_text("Confirm this synthetic environment for this test.\n")
        self.cli(
            "confirm",
            "--fingerprint",
            environment.fingerprint(self.report),
            "--user-response-file",
            reply,
        )

    def task(self, key="op", kind="static"):
        self.cli(
            "task-add",
            "--id",
            key,
            "--title",
            key,
            "--files",
            f"vllm_metax/{key}.py",
            "--requires",
            kind,
            "--configuration",
            "fixture",
            "--next-action",
            "Run fixture validation",
        )
        synthetic_mapping(self.run, key)

    def check(self, key="op", kind="static", code="pass", script=None):
        command = (
            [sys.executable, str(script)] if script else [sys.executable, "-c", code]
        )
        return self.cli(
            "run-check",
            "--id",
            key,
            "--kind",
            kind,
            "--configuration",
            "fixture",
            "--execution-scope",
            "model"
            if kind in {"e2e", "batched_test", "weight_loading", "distributed", "graph"}
            else "component",
            "--",
            *command,
        )

    def done(self, key="op"):
        state = store.load(self.run)
        if (
            state["inputs"]["mode"] == "full"
            and state["tasks"][key]["status"] != "done"
        ):
            self.check(key, "static")
            self.check(key, "review")
            self.cli(
                "task-set",
                "--id",
                key,
                "--status",
                "implemented",
                "--note",
                "Refresh source checks after production failure",
            )
        self.cli("task-set", "--id", key, "--status", "done")

    def put(self, action, document, key=None):
        path = self.root / (action + ".json")
        path.write_text(json.dumps(document))
        return self.cli(action, "--file", path, *(["--id", key] if key else []))

    def report_code(self, configuration):
        report = {
            "vllm": str(self.vllm / "vllm/__init__.py"),
            "vllm_metax": str(self.metax / "vllm_metax/__init__.py"),
            "dispatch": {"backend": "synthetic-backend"},
            "configuration": configuration,
        }
        artifacts = []
        for name in ("result.json", "inference_results.manifest.json"):
            path = self.run / name
            payload = {"synthetic_fixture": True}
            if name.endswith(".manifest.json"):
                payload = {
                    "schema_version": 2,
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
        report["batched_test"] = {
            "cases": [{"id": "fixture", "status": "passed", "correct_ratio": 1.0}],
            "artifacts": artifacts,
        }
        return (
            "import os,json; from pathlib import Path; "
            "Path(os.environ['METAX_UPGRADE_ORIGINS_FILE']).write_text("
            + repr(json.dumps(report))
            + ")"
        )

    def test_cli_archive_interruption_recovers_via_refresh_and_resume(self):
        original_write = zipfile.ZipFile.write

        def interrupted(archive, *args, **kwargs):
            original_write(archive, *args, **kwargs)
            raise InterruptedError("injected initialization interruption")

        for action in ("refresh", "resume"):
            with self.subTest(action=action):
                run = self.root / ("interrupted-" + action)
                with patch.object(zipfile.ZipFile, "write", interrupted):
                    with self.assertRaises(InterruptedError):
                        self.init(run)
                self.assertEqual(store.load(run)["initialization"]["phase"], "baseline")
                self.assertTrue(list((run / "baseline").glob("*.partial")))
                if action == "resume":
                    with self.assertRaisesRegex(ValueError, "not user-confirmed"):
                        self.cli(action, run=run)
                else:
                    self.cli(action, run=run)
                recovered = store.load(run)
                self.assertEqual(recovered["initialization"]["phase"], "ready")
                self.assertTrue(recovered["environment"]["full_collection"])
                self.assertTrue(
                    lifecycle._archive_valid(
                        run / "baseline/metax-source.zip", recovered["source_manifest"]
                    )
                )

    def test_old_orphan_initialization_preserves_and_recovers_partial_archive(self):
        run = self.root / "orphan"
        folder = run / "baseline"
        store.write_json(
            folder / "source-manifest.json", store.load(self.run)["source_manifest"]
        )
        (folder / "metax-source.zip").write_bytes(b"old interrupted initialization")
        self.init(run)
        recovered = store.load(run)
        self.assertEqual(recovered["initialization"]["phase"], "ready")
        preserved = list(folder.glob("metax-source.interrupted-*.zip"))
        self.assertEqual(len(preserved), 1)
        self.assertEqual(preserved[0].read_bytes(), b"old interrupted initialization")

    def test_legacy_migration_preserves_original_and_invalidates_old_evidence(self):
        self.task()
        self.assertEqual(self.check(), 0)
        self.done()
        state = store.load(self.run)
        state["schema_version"] = 1
        for field in (
            "policy_identity",
            "policy_history",
            "initialization",
            "support_matrix",
        ):
            state.pop(field, None)
        state["tasks"]["op"].pop("upstream_mapping")
        for observation in state["tasks"]["op"]["evidence"]:
            observation.pop("input_hashes", None)
            observation.pop("input_files", None)
        store.save(self.run, state)
        original = (self.run / "state.json").read_bytes()
        with self.assertRaisesRegex(ValueError, "Legacy run"):
            self.check()
        self.cli(
            "migrate", "--note", "Reviewed legacy acceptance and will rerun validation"
        )
        backups = list((self.run / "migrations").glob("schema-1-*.json"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_bytes(), original)
        migrated = store.load(self.run)
        self.assertEqual(migrated["schema_version"], store.SCHEMA_VERSION)
        self.assertEqual(migrated["tasks"]["op"]["status"], "implemented")
        self.assertTrue(migrated["tasks"]["op"]["evidence"][0]["stale"])
        self.assertEqual(store.missing_evidence(migrated, "op"), ["static"])

    def test_policy_change_requires_review_and_fresh_validation(self):
        self.task()
        self.assertEqual(self.check(), 0)
        self.done()
        changed = copy.deepcopy(lifecycle.runtime_identity())
        changed["files"]["synthetic-rule.md"] = "1" * 64
        changed["fingerprint"] = store.digest(changed["files"])
        with patch.object(lifecycle, "runtime_identity", return_value=changed):
            with self.assertRaisesRegex(ValueError, "accept-policy-change"):
                self.check()
            self.cli(
                "accept-policy-change",
                "--note",
                "Reviewed the changed synthetic acceptance rule",
            )
            self.assertTrue(store.load(self.run)["tasks"]["op"]["evidence"][0]["stale"])
            with self.assertRaisesRegex(ValueError, "Missing current"):
                self.done()
            self.assertEqual(self.check(), 0)
            self.done()
            observations = store.load(self.run)["tasks"]["op"]["evidence"]
            self.assertEqual(len(observations), 2)
            self.assertFalse(observations[-1]["stale"])

    def test_external_test_script_change_invalidates_completed_evidence(self):
        self.task()
        script = self.root / "external-verifier.py"
        script.write_text("print('first validation')\n")
        self.assertEqual(self.check(script=script), 0)
        self.done()
        original = store.load(self.run)["tasks"]["op"]["evidence"][-1]
        self.assertIn(str(script), original["input_hashes"])
        script.write_text("print('changed validation')\n")
        self.cli("status")
        invalidated = store.load(self.run)["tasks"]["op"]
        self.assertEqual(invalidated["status"], "implemented")
        self.assertTrue(invalidated["evidence"][-1]["stale"])
        with self.assertRaisesRegex(ValueError, "Missing current"):
            self.done()
        self.assertEqual(self.check(script=script), 0)
        self.done()

    def test_numeric_policy_change_needs_reason_and_preserves_old_contract(self):
        self.task()
        contract = {
            "reference": {"description": "Independent reference", "revision": "v1"},
            "seed": 17,
            "tolerances": {"atol": 0.001, "rtol": 0.01},
            "nan_policy": "forbid",
            "inf_policy": "forbid",
            "cases": ["boundary"],
        }
        self.put("numeric-put", contract, "op")
        self.assertEqual(self.check(), 0)
        self.done()
        changed = copy.deepcopy(contract)
        changed["tolerances"]["atol"] = 0.002
        with self.assertRaisesRegex(ValueError, "technical reason"):
            self.put("numeric-put", changed, "op")
        self.assertEqual(
            store.load(self.run)["tasks"]["op"]["numerical_contract"], contract
        )
        self.cli(
            "numeric-put",
            "--id",
            "op",
            "--file",
            self.root / "numeric-put.json",
            "--note",
            "Revised bound after independently checking reference rounding",
        )
        current = store.load(self.run)
        self.assertEqual(current["acceptance_history"][-1]["previous"], contract)
        self.assertEqual(current["acceptance_history"][-1]["current"], changed)
        self.assertTrue(current["tasks"]["op"]["evidence"][-1]["stale"])
        self.assertEqual(current["tasks"]["op"]["status"], "implemented")

    def test_explicit_data_input_is_relative_to_runtime_and_invalidates_evidence(self):
        self.task()
        data = self.root / "indirect-input.json"
        data.write_text('{"value": 1}\n')
        script = self.root / "indirect-data-test.py"
        script.write_text(
            "from pathlib import Path; print(Path('indirect-input.json').read_text())\n"
        )
        self.assertEqual(
            self.cli(
                "run-check",
                "--id",
                "op",
                "--kind",
                "static",
                "--configuration",
                "fixture",
                "--input-file",
                data.name,
                "--",
                sys.executable,
                str(script),
            ),
            0,
        )
        self.done()
        evidence = store.load(self.run)["tasks"]["op"]["evidence"][-1]
        self.assertIn(str(data), evidence["input_hashes"])
        data.write_text('{"value": 2}\n')
        self.cli("status")
        invalidated = store.load(self.run)["tasks"]["op"]
        self.assertEqual(invalidated["status"], "implemented")
        self.assertTrue(invalidated["evidence"][-1]["stale"])

    def test_explicit_symlink_input_retargeting_invalidates_evidence(self):
        self.task()
        first, second = self.root / "first.json", self.root / "second.json"
        first.write_text('{"value": 1}\n')
        second.write_text('{"value": 2}\n')
        selected = self.root / "selected.json"
        selected.symlink_to(first)
        self.assertEqual(
            self.cli(
                "run-check",
                "--id",
                "op",
                "--kind",
                "static",
                "--configuration",
                "fixture",
                "--input-file",
                selected.name,
                "--",
                sys.executable,
                "-c",
                "pass",
            ),
            0,
        )
        self.done()
        selected.unlink()
        selected.symlink_to(second)
        with self.assertRaisesRegex(ValueError, "Missing current"):
            self.done()
        persisted = store.load(self.run)["tasks"]["op"]
        self.assertEqual(persisted["status"], "implemented")
        self.assertTrue(persisted["evidence"][-1]["stale"])

    def test_full_matrix_requires_each_row_e2e_and_actual_configuration(self):
        self.task(kind="e2e")
        self.task("second", kind="static")
        state = store.load(self.run)
        state["inputs"]["mode"] = "full"
        state["tasks"]["op"]["files"] = list(state["source_manifest"])
        store.save(self.run, state)
        synthetic_mapping(self.run, "op")
        configurations = {
            "op": {
                "model": "fixture-model",
                "dtype": "bf16",
                "tensor_parallel_size": 1,
            },
            "second": {
                "model": "fixture-model",
                "dtype": "fp16",
                "tensor_parallel_size": 2,
            },
        }
        self.put(
            "matrix-put",
            {
                "rows": [
                    {
                        "id": key,
                        "status": "supported",
                        "configuration": config,
                        "tasks": [key],
                        "requires": ["e2e"],
                        "dispatch": {"backend": "synthetic-backend"},
                    }
                    for key, config in configurations.items()
                ]
            },
        )
        prepare_full_validation(self.cli, self.run)
        self.assertEqual(
            self.check(kind="e2e", code=self.report_code(configurations["op"])), 0
        )
        self.assertEqual(
            self.check(
                kind="batched_test", code=self.report_code(configurations["op"])
            ),
            0,
        )
        self.done()
        self.assertEqual(self.check("second"), 0)
        self.assertEqual(
            self.check(
                "second", "batched_test", self.report_code(configurations["second"])
            ),
            0,
        )
        self.done("second")
        with self.assertRaisesRegex(ValueError, "second: missing current e2e"):
            self.cli("finish")
        self.assertEqual(
            self.check("second", "e2e", self.report_code(configurations["op"])), 1
        )
        failed = store.load(self.run)["tasks"]["second"]["evidence"][-1]
        self.assertIn("Actual configuration differs", failed["origins"])
        self.assertEqual(self.check("second"), 0)
        self.assertEqual(self.check("second", "review"), 0)
        self.cli(
            "task-set",
            "--id",
            "second",
            "--status",
            "implemented",
            "--note",
            "Review source after failed configuration acceptance",
        )
        self.assertEqual(
            self.check("second", "e2e", self.report_code(configurations["second"])), 0
        )
        # A failed production check invalidates the previously completed task,
        # including its earlier static evidence.
        self.assertEqual(self.check("second"), 0)
        self.assertEqual(
            self.check(
                "second", "batched_test", self.report_code(configurations["second"])
            ),
            0,
        )
        self.done("second")
        self.cli("finish")
        self.assertIsNotNone(store.load(self.run)["completed_at"])


if __name__ == "__main__":
    unittest.main()
