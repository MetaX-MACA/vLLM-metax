# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Offline acceptance gates reject incomplete coverage and unmeasured claims."""

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from metax_upgrade import acceptance, lifecycle, state as store


class AcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.metax = self.root / "metax"
        self.upstream = self.root / "upstream"
        self.metax.mkdir()
        self.upstream.mkdir()
        (self.metax / "op.py").write_text("def op(value):\n    return value + 1\n")
        (self.upstream / "op.py").write_text("def op(value):\n    return value\n")
        self.state = {
            "inputs": {
                "metax_source": str(self.metax),
                "vllm_source": str(self.upstream),
                "mode": "full",
            },
            "tasks": {},
            "workflow": {"phase": "validation", "history": [], "unverified": []},
            "inventory": {},
            "dependency_analysis": {
                "files": ["op.py"],
                "models": [],
                "edges": [],
                "cycles": [],
                "summary": "Synthetic isolated function",
                "environment": "confirmed",
            },
            "contracts": {},
            "environment": {"fingerprint": "confirmed"},
        }
        self.add_task("op")
        self.mapping = {
            "mappings": [
                {
                    "local": "op.py",
                    "upstream": {
                        "revision": "v1",
                        "path": "op.py",
                        "symbol": "op",
                        "sha256": hashlib.sha256(
                            (self.upstream / "op.py").read_bytes()
                        ).hexdigest(),
                    },
                    "previous": {
                        "absent": True,
                        "reason": "Newly introduced adaptation",
                    },
                    "disposition": "update",
                    "metax_semantics": "Preserve platform adjustment",
                    "reason": "Target API changed",
                    "removal_condition": "Remove when upstream supports platform",
                }
            ],
        }
        acceptance.put_mapping(self.state, "op", self.mapping)
        self.row = {
            "id": "bf16",
            "status": "supported",
            "configuration": {
                "model": "Example",
                "dtype": "bf16",
                "parallel": {"tp": 1},
            },
            "requires": ["e2e"],
            "tasks": ["op"],
        }
        acceptance.put_matrix(self.state, {"rows": [self.row]})
        self.observation = {
            "configuration": copy.deepcopy(self.row["configuration"]),
            "dispatch": {"attention": "metax"},
        }
        self.numeric = {
            "reference": {
                "description": "Independent mathematical reference",
                "revision": "reference-v1",
            },
            "seed": 17,
            "tolerances": {"atol": 0.001, "rtol": 0.01},
            "nan_policy": "forbid",
            "inf_policy": "forbid",
            "cases": ["small", "boundary"],
        }

    def add_task(self, key, kinds=None):
        self.state["tasks"][key] = {
            "files": ["op.py"],
            "configuration": key,
            "requires": kinds or ["static"],
            "depends_on": [],
            "evidence": [],
            "status": "done",
            "local_checks": {
                "changes": ["none"],
                "reason": "Synthetic acceptance fixture",
                "required": [],
            },
        }

    def evidence(self, key="op", kind="e2e", report=None, **changes):
        self.state["tasks"][key]["source_signature"] = store.source_signature(
            self.state, key
        )
        if kind == "batched_test" and report is None:
            report = copy.deepcopy(self.observation)
            artifacts = []
            for name in ("result.json", "inference_results.manifest.json"):
                path = self.root / name
                payload = {"synthetic_fixture": True}
                if name.endswith(".manifest.json"):
                    payload = {
                        "schema_version": 2,
                        "validation_identities": {
                            "fixture": {
                                **store.validation_identity(self.state),
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
        evidence = {
            "kind": kind,
            "result": "passed",
            "signature": store.check_signature(self.state, key, kind),
            "environment": "confirmed",
            "stale": False,
            "report": copy.deepcopy(self.observation if report is None else report),
            "command": [sys.executable, "-c", "pass"],
            "cwd": str(self.root),
            "input_hashes": lifecycle.evidence_inputs(
                [sys.executable, "-c", "pass"], self.root
            ),
        }
        evidence.update(changes)
        self.state["tasks"][key]["evidence"].append(evidence)
        return evidence

    def numerical_report(self):
        report = copy.deepcopy(self.observation)
        report["numerical"] = {
            "contract_digest": store.digest(self.numeric),
            "cases": [
                {
                    "id": key,
                    "max_abs": 0.005,
                    "reference_scale": 1.0,
                    "nan_count": 0,
                    "inf_count": 0,
                    "passed": True,
                }
                for key in self.numeric["cases"]
            ],
        }
        return report

    def test_each_full_support_row_needs_its_own_e2e(self):
        self.state["tasks"]["op"]["requires"] = ["e2e"]
        self.evidence()
        self.evidence(kind="batched_test")
        self.assertEqual(acceptance.validate_finish(self.state), [])
        self.add_task("fp32", ["static"])
        acceptance.put_mapping(self.state, "fp32", self.mapping)
        other = {
            **self.row,
            "id": "fp32",
            "tasks": ["fp32"],
            "configuration": {"dtype": "fp32"},
            "requires": ["static"],
        }
        acceptance.put_matrix(self.state, {"rows": [self.row, other]})
        # Adding another row does not let the first row's e2e cover its configuration.
        self.evidence()
        self.evidence(key="fp32", kind="static", report={})
        self.assertTrue(
            any(
                "fp32: missing current e2e" in error
                for error in acceptance.validate_finish(self.state)
            )
        )

    def test_latest_failure_and_stale_or_wrong_environment_cannot_cover_matrix(self):
        for changes in (
            {"result": "failed"},
            {"stale": True},
            {"environment": "other"},
            {"signature": "old"},
        ):
            self.state["tasks"]["op"]["evidence"] = []
            self.evidence(**changes)
            self.assertTrue(
                any(
                    "missing current e2e" in error
                    for error in acceptance.validate_finish(self.state)
                )
            )
        self.state["tasks"]["op"]["evidence"] = []
        self.evidence()
        self.evidence(result="failed")
        self.assertTrue(
            any(
                "missing current e2e" in error
                for error in acceptance.validate_finish(self.state)
            )
        )

    def test_changed_validation_input_invalidates_matrix_evidence(self):
        self.state["tasks"]["op"]["requires"] = ["e2e"]
        script = self.root / "check.py"
        script.write_text("# original validation\n")
        command = [sys.executable, str(script)]
        self.evidence(
            command=command, input_hashes=lifecycle.evidence_inputs(command, self.root)
        )
        self.evidence(kind="batched_test")
        self.assertEqual(acceptance.validate_finish(self.state), [])
        script.write_text("# changed assertions\n")
        self.assertTrue(
            any(
                "missing current e2e" in error
                for error in acceptance.validate_finish(self.state)
            )
        )

    def test_declaration_preflight_does_not_require_evidence_it_will_generate(self):
        self.assertEqual(
            acceptance.validate_task(self.state, "op", require_evidence=False), []
        )
        self.assertTrue(acceptance.validate_task(self.state, "op"))

    def test_expected_dispatch_and_recursive_observed_configuration(self):
        changed = copy.deepcopy(self.observation)
        changed["configuration"]["parallel"]["tp"] = 2
        self.assertTrue(acceptance.validate_report(self.state, "op", "e2e", changed))
        self.row["dispatch"] = {"attention": "metax"}
        acceptance.put_matrix(self.state, [self.row])
        self.assertEqual(
            acceptance.validate_report(self.state, "op", "e2e", self.observation), []
        )
        changed = copy.deepcopy(self.observation)
        changed["dispatch"] = {"attention": "fallback"}
        self.assertTrue(
            any(
                "dispatch" in error
                for error in acceptance.validate_report(
                    self.state, "op", "e2e", changed
                )
            )
        )
        self.assertTrue(
            acceptance.validate_report(
                self.state, "op", "e2e", {"configuration": "bf16"}
            )
        )

    def test_ambiguous_task_configurations_and_empty_matrix_are_rejected(self):
        other = {**self.row, "id": "other", "configuration": {"dtype": "fp32"}}
        with self.assertRaisesRegex(ValueError, "conflicting"):
            acceptance.put_matrix(self.state, [self.row, other])
        with self.assertRaisesRegex(ValueError, "nonempty"):
            acceptance.put_matrix(self.state, [])
        del self.state["support_matrix"]
        self.assertTrue(
            acceptance.validate_report(self.state, "op", "e2e", self.observation)
        )
        self.assertTrue(
            any(
                "support matrix" in error
                for error in acceptance.validate_finish(self.state)
            )
        )

    def test_unsupported_rows_require_actual_rejection_evidence(self):
        row = {
            **self.row,
            "status": "unsupported",
            "rejection_reason": "Unsupported architecture",
            "requires": ["contract"],
        }
        acceptance.put_matrix(self.state, [row])
        self.assertTrue(
            acceptance.validate_report(self.state, "op", "contract", self.observation)
        )
        rejected = {
            **self.observation,
            "rejected": True,
            "rejection_reason": "Backend rejects this architecture",
        }
        self.assertEqual(
            acceptance.validate_report(self.state, "op", "contract", rejected), []
        )
        self.evidence(kind="static")
        self.evidence(kind="contract", report=rejected)
        self.assertTrue(
            any(
                "at least one supported row" in error
                for error in acceptance.validate_finish(self.state)
            )
        )
        self.state["inputs"]["mode"] = "component"
        self.assertEqual(acceptance.validate_finish(self.state), [])
        rejected["configuration"] = {"dtype": "fp32"}
        self.assertTrue(
            acceptance.validate_report(self.state, "op", "contract", rejected)
        )

    def test_supported_rows_do_not_accept_rejection_as_pass(self):
        report = {
            **self.observation,
            "rejected": True,
            "rejection_reason": "Unexpected failure",
        }
        self.assertTrue(acceptance.validate_report(self.state, "op", "e2e", report))

    def test_numeric_requires_contract_digest_complete_cases_and_measured_tolerance(
        self,
    ):
        self.assertTrue(
            acceptance.validate_report(
                self.state, "op", "component_numeric", self.observation
            )
        )
        acceptance.put_numeric(self.state, "op", self.numeric)
        report = self.numerical_report()
        self.assertEqual(
            acceptance.validate_report(self.state, "op", "component_numeric", report),
            [],
        )
        for mutation in (
            lambda r: r["numerical"].update(contract_digest="different"),
            lambda r: r["numerical"]["cases"].pop(),
            lambda r: r["numerical"]["cases"][0].update(max_abs=0.1),
            lambda r: r["numerical"]["cases"][0].update(max_abs=float("nan")),
            lambda r: r["numerical"]["cases"][0].update(passed=False),
            lambda r: r["numerical"]["cases"][0].update(nan_count=1),
            lambda r: r["numerical"]["cases"][0].update(inf_count=1),
        ):
            bad = copy.deepcopy(report)
            mutation(bad)
            self.assertTrue(
                acceptance.validate_report(self.state, "op", "component_numeric", bad)
            )

    def test_matching_nonfinite_policies_need_reference_pattern_comparison(self):
        self.numeric.update(nan_policy="match_reference", inf_policy="match_reference")
        acceptance.put_numeric(self.state, "op", self.numeric)
        report = self.numerical_report()
        self.assertTrue(
            acceptance.validate_report(self.state, "op", "component_numeric", report)
        )
        for case in report["numerical"]["cases"]:
            case.update(
                nan_count=1,
                inf_count=2,
                nan_matches_reference=True,
                inf_matches_reference=True,
            )
        self.assertEqual(
            acceptance.validate_report(self.state, "op", "component_numeric", report),
            [],
        )

    def test_declared_numeric_contract_applies_to_production_checks(self):
        acceptance.put_numeric(self.state, "op", self.numeric)
        self.assertTrue(
            acceptance.validate_report(self.state, "op", "e2e", self.observation)
        )
        self.assertEqual(
            acceptance.validate_report(
                self.state, "op", "e2e", self.numerical_report()
            ),
            [],
        )

    def test_numeric_definition_changes_identity_and_rejects_invalid_contracts(self):
        before = copy.deepcopy(acceptance.task_identity(self.state, "op"))
        acceptance.put_numeric(self.state, "op", self.numeric)
        self.assertNotEqual(before, acceptance.task_identity(self.state, "op"))
        for change in (
            {"seed": True},
            {"tolerances": {"atol": -1, "rtol": 0}},
            {"cases": ["same", "same"]},
            {"reference": {"description": "missing revision"}},
        ):
            with self.assertRaises(ValueError):
                acceptance.put_numeric(self.state, "op", {**self.numeric, **change})

    def test_mapping_requires_all_files_and_exact_target_hash_and_symbol(self):
        for change in (
            {"sha256": "0" * 64},
            {"symbol": "missing"},
            {"path": "../op.py"},
        ):
            mapping = copy.deepcopy(self.mapping)
            mapping["mappings"][0]["upstream"].update(change)
            with self.assertRaises(ValueError):
                acceptance.put_mapping(self.state, "op", mapping)
        (self.metax / "other.py").write_text("pass\n")
        self.state["tasks"]["op"]["files"].append("other.py")
        with self.assertRaisesRegex(ValueError, "lack upstream mappings"):
            acceptance.put_mapping(self.state, "op", self.mapping)
        self.assertTrue(
            any(
                "lack upstream mappings" in error
                for error in acceptance.validate_task(self.state, "op")
            )
        )

    def test_mapping_accepts_explicit_platform_only_and_deleted_dispositions(self):
        mapping = copy.deepcopy(self.mapping)
        item = mapping["mappings"][0]
        item.update(
            disposition="platform_only",
            upstream={"absent": True, "reason": "Vendor-specific integration"},
        )
        acceptance.put_mapping(self.state, "op", mapping)
        (self.metax / "op.py").unlink()
        with self.assertRaisesRegex(ValueError, "remove disposition"):
            acceptance.put_mapping(self.state, "op", mapping)
        item["disposition"] = "remove"
        acceptance.put_mapping(self.state, "op", mapping)
        self.evidence(kind="static")
        self.assertEqual(acceptance.validate_task(self.state, "op"), [])

    def test_current_mapping_rejects_changed_target_source(self):
        (self.upstream / "op.py").write_text("def op(value):\n    return -value\n")
        self.evidence(kind="static")
        self.assertTrue(
            any(
                "sha256" in error
                for error in acceptance.validate_task(self.state, "op")
            )
        )

    def test_git_target_revision_must_match_selected_revision_and_clean_blob(self):
        def git(*args):
            return (
                subprocess.run(
                    ["git", "-C", str(self.upstream), *args],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                .stdout.decode()
                .strip()
            )

        git("init", "-q")
        git("add", "op.py")
        git(
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "base",
        )
        base = git("rev-parse", "HEAD")
        (self.upstream / "new.py").write_text("pass\n")
        git("add", "new.py")
        git(
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "target",
        )
        mapping = copy.deepcopy(self.mapping)
        mapping["mappings"][0]["upstream"]["revision"] = base
        with self.assertRaisesRegex(ValueError, "differs from the selected target"):
            acceptance.put_mapping(self.state, "op", mapping)
        mapping["mappings"][0]["upstream"]["revision"] = "HEAD"
        acceptance.put_mapping(self.state, "op", mapping)
        (self.upstream / "op.py").write_text("def op(value):\n    return 7\n")
        with self.assertRaisesRegex(ValueError, "working-tree"):
            acceptance.put_mapping(self.state, "op", mapping)
        git("checkout", "--", "op.py")
        (self.upstream / "new.py").write_text("# another target revision\n")
        git("add", "new.py")
        git(
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "new target",
        )
        self.assertTrue(
            any(
                "revision moved" in issue
                for issue in acceptance.validate_task(
                    self.state, "op", require_evidence=False
                )
            )
        )

    def test_review_artifacts_preserve_old_diffs_after_source_changes(self):
        run = self.root / "run"
        acceptance.refresh_artifacts(run, self.state)
        first = self.state["tasks"]["op"]["review_artifact"]
        record = json.loads((run / first).read_text())
        diff = (run / first).parent / record["mappings"][0]["diff"]
        original = diff.read_text()
        self.assertIn("+    return value + 1", original)
        (self.metax / "op.py").write_text("def op(value):\n    return value + 2\n")
        acceptance.refresh_artifacts(run, self.state)
        self.assertNotEqual(first, self.state["tasks"]["op"]["review_artifact"])
        self.assertEqual(diff.read_text(), original)


if __name__ == "__main__":
    unittest.main()
