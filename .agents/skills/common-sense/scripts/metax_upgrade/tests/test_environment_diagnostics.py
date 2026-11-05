# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Offline regression checks for fresh, supplementary collect_env diagnostics."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from metax_upgrade import environment


class CollectorDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.script = self.root / "vllm_metax/collect_env.py"
        self.script.parent.mkdir()
        self.script.write_text("import uuid\nprint(uuid.uuid4())\n")

    def collect(self):
        return environment._collect_env(self.root, self.root, 5)

    def test_same_script_runs_fresh_without_writing_files(self):
        before = sorted(self.root.rglob("*"))
        first, second = self.collect(), self.collect()
        self.assertEqual(first["status"], "ok")
        self.assertEqual(second["status"], "ok")
        self.assertNotEqual(first["stdout"], second["stdout"])
        self.assertEqual(first["source_hash"], second["source_hash"])
        self.assertTrue(first["collected_at"])
        self.assertEqual(before, sorted(self.root.rglob("*")))
        self.assertFalse(any(key.startswith("cache") for key in second))
        self.assertEqual(first["command"], [sys.executable, "-B", str(self.script)])

    def test_changed_source_is_collected_and_plain_formatter_keeps_output(self):
        first = self.collect()
        self.script.write_text("print('updated report')\n")
        updated = self.collect()
        self.assertNotEqual(first["source_hash"], updated["source_hash"])
        self.assertEqual(updated["stdout"], "updated report\n")
        self.assertEqual(
            environment.format_environment({"details": {"collect_env": updated}}),
            "updated report",
        )

    def test_failures_run_again_and_preserve_diagnostics(self):
        self.script.write_text(
            "import sys\nprint('failed', file=sys.stderr)\nsys.exit(1)\n"
        )
        with patch.object(environment, "_run", wraps=environment._run) as run:
            first, second = self.collect(), self.collect()
        self.assertEqual(run.call_count, 2)
        self.assertEqual(first["status"], "error")
        self.assertEqual(second["status"], "error")
        self.assertEqual(second["returncode"], 1)
        self.assertIn("failed", second["stderr"])

    def test_missing_collector_is_diagnostic_and_does_not_run_subprocess(self):
        self.script.unlink()
        with patch.object(environment, "_run") as run:
            missing = self.collect()
        run.assert_not_called()
        self.assertEqual(missing["status"], "missing")
        self.assertEqual(missing["source"], str(self.script))
        self.assertIsNone(missing["source_hash"])
        self.assertTrue(missing["collected_at"])
        self.assertTrue(missing["stderr"])

    def test_supplementary_child_disables_plugins_and_bytecode(self):
        result = {"status": "ok", "returncode": 0, "stdout": "report", "stderr": ""}
        with patch.object(environment, "_run", return_value=result) as run:
            self.collect()
        child_env = run.call_args.args[3]
        self.assertEqual(child_env["VLLM_PLUGINS"], "")
        self.assertEqual(child_env["PYTHONDONTWRITEBYTECODE"], "1")

    def test_identity_rechecks_recollect_facts_without_supplementary_child(self):
        requirements = self.root / "requirements.txt"
        requirements.write_text("")
        result = {"status": "ok", "returncode": 0, "stdout": "{}", "stderr": ""}
        versions = [
            ([{"normalized_name": "torch", "version": version}], [])
            for version in ("1", "2")
        ]
        with (
            patch.object(environment, "_inventory", side_effect=versions) as inventory,
            patch.object(environment, "_run", return_value=result) as probes,
            patch.object(environment, "_collect_env") as collect,
        ):
            reports = [
                environment.collect_environment(
                    self.root,
                    self.root,
                    self.root,
                    requirements=requirements,
                    include_collect_env=False,
                )
                for _ in range(2)
            ]
        collect.assert_not_called()
        self.assertEqual(inventory.call_count, 2)
        self.assertGreater(probes.call_count, 0)
        self.assertNotEqual(
            environment.fingerprint(reports[0]), environment.fingerprint(reports[1])
        )
        self.assertEqual(reports[1]["details"]["collection_profile"], "identity")
        self.assertEqual(reports[1]["details"]["collect_env"]["status"], "skipped")


if __name__ == "__main__":
    unittest.main()
