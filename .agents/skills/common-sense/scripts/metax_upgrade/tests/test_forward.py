# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Forward scenarios using real durable state and child processes, without GPUs."""

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from metax_upgrade import state, upgrade
from metax_upgrade.tests.test_upgrade import synthetic_mapping


class ForwardScenarios(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.metax = self.root / "metax"
        self.vllm = self.root / "upstream"
        self.run = self.root / "run"
        for directory in (self.metax / "vllm_metax", self.vllm / "vllm"):
            directory.mkdir(parents=True)
            (directory / "__init__.py").write_text("")
        (self.metax / "vllm_metax/producer.py").write_text("VALUE = 1\n")
        (self.metax / "vllm_metax/consumer.py").write_text("VALUE = 1\n")
        (self.metax / "requirements").mkdir()
        (self.metax / "requirements/maca_private.txt").write_text("kernel==1\n")
        self.report = {
            "identity": {
                "sdk": {"sdk_versions": ["3.1"]},
                "gpu": {"models": [{"index": 0, "description": "MXC-test"}]},
                "packages": [{"name": "kernel", "version": "1"}],
                "source_mapping": {
                    "vllm": str(self.vllm),
                    "vllm_metax": str(self.metax),
                },
            },
            "readiness": {"issues": []},
        }
        self.collector = patch.object(
            upgrade, "collect", side_effect=lambda *_a, **_k: copy.deepcopy(self.report)
        )
        self.collector.start()
        self.addCleanup(self.collector.stop)
        self.call(
            "init",
            "--vllm-source",
            self.vllm,
            "--metax-source",
            self.metax,
            "--runtime-cwd",
            self.root,
            "--scope",
            "synthetic recovery scenario",
            "--mode",
            "component",
        )

    def call(self, action, *args):
        with contextlib.redirect_stdout(io.StringIO()):
            try:
                return upgrade.main(
                    [action, "--run-dir", str(self.run), *map(str, args)]
                )
            finally:
                if action == "init" and (self.run / "state.json").exists():
                    legacy = state.load(self.run)
                    legacy["inputs"].pop("environment_protocol", None)
                    state.save(self.run, legacy)

    def read(self):
        return state.load(self.run)

    def confirm(self, acknowledge=False):
        response = self.root / "actual-user-response.txt"
        response.write_text(
            "Confirm the disclosed synthetic environment for this test.\n"
        )
        args = [
            "--fingerprint",
            self.read()["environment"]["fingerprint"],
            "--user-response-file",
            response,
        ]
        if acknowledge:
            args.append("--acknowledge-issues")
        self.call("confirm", *args)

    def task(self, key, dependencies=(), requires=("static",)):
        args = [
            "--id",
            key,
            "--title",
            key,
            "--files",
            f"vllm_metax/{key}.py",
            "--configuration",
            "test-config",
            "--next-action",
            "Review the target implementation",
        ]
        for dependency in dependencies:
            args.extend(["--depends-on", dependency])
        for kind in requires:
            args.extend(["--requires", kind])
        self.call("task-add", *args)
        synthetic_mapping(self.run, key)

    def check(self, key, code="pass", kind="static"):
        return self.call(
            "run-check",
            "--id",
            key,
            "--kind",
            kind,
            "--configuration",
            "test-config",
            "--timeout",
            "10",
            "--",
            sys.executable,
            "-c",
            code,
        )

    def done(self, key):
        self.assertEqual(self.check(key), 0)
        self.call("task-set", "--id", key, "--status", "done")

    def test_pending_user_confirmation_survives_new_agent(self):
        self.task("producer")
        with self.assertRaisesRegex(ValueError, "not user-confirmed"):
            self.call("task-set", "--id", "producer", "--status", "in_progress")
        with self.assertRaisesRegex(ValueError, "not user-confirmed"):
            self.check("producer")
        with self.assertRaisesRegex(ValueError, "not user-confirmed"):
            self.call("resume")
        self.assertIn("WAITING FOR USER", (self.run / "TODO.md").read_text())
        self.confirm()
        self.call("resume")
        self.call("task-set", "--id", "producer", "--status", "in_progress")
        self.assertEqual(self.read()["tasks"]["producer"]["status"], "in_progress")

    def test_source_and_contract_changes_invalidate_transitive_consumers(self):
        self.confirm()
        self.task("producer")
        self.task("consumer", ["producer"])
        self.done("producer")
        self.done("consumer")
        (self.metax / "vllm_metax/producer.py").write_text("VALUE = 2\n")
        self.call("resume")
        changed = self.read()
        for key in ("producer", "consumer"):
            self.assertEqual(changed["tasks"][key]["status"], "implemented")
            self.assertTrue(all(e["stale"] for e in changed["tasks"][key]["evidence"]))
        self.done("producer")
        self.done("consumer")
        contract = self.root / "contract.json"
        contract.write_text(
            json.dumps(
                {
                    "id": "layout",
                    "upstream": "vllm/source.py:Function",
                    "summary": "layout semantics",
                    "invariants": ["units are blocks"],
                    "producer": "producer",
                    "consumers": ["consumer"],
                }
            )
        )
        self.call("contract-put", "--file", contract)
        for key in ("producer", "consumer"):
            self.assertEqual(self.read()["tasks"][key]["status"], "implemented")
            self.assertTrue(state.missing_evidence(self.read(), key))

    def test_latest_failure_cannot_hide_behind_previous_pass(self):
        self.confirm()
        self.task("producer")
        self.done("producer")
        self.assertEqual(self.check("producer", "raise SystemExit(3)"), 1)
        self.assertEqual(self.read()["tasks"]["producer"]["status"], "implemented")
        with self.assertRaisesRegex(ValueError, "Missing current"):
            self.call("task-set", "--id", "producer", "--status", "done")
        self.assertEqual(len(self.read()["tasks"]["producer"]["evidence"]), 2)

    def test_failed_completion_still_persists_detected_source_invalidation(self):
        self.confirm()
        self.task("producer")
        self.done("producer")
        (self.metax / "vllm_metax/producer.py").write_text("VALUE = 2\n")
        with self.assertRaisesRegex(ValueError, "Missing current"):
            self.call("task-set", "--id", "producer", "--status", "done")
        persisted = self.read()
        self.assertEqual(persisted["tasks"]["producer"]["status"], "implemented")
        self.assertTrue(persisted["tasks"]["producer"]["evidence"][-1]["stale"])
        self.assertIn("[implemented]", (self.run / "TODO.md").read_text())

    def test_environment_drift_revokes_approval_and_prior_evidence(self):
        self.confirm()
        self.task("producer")
        self.done("producer")
        self.report["identity"]["sdk"]["sdk_versions"] = ["3.2"]
        with self.assertRaisesRegex(ValueError, "Environment drift"):
            self.call("resume")
        changed = self.read()
        self.assertIsNone(changed["environment"]["confirmation"])
        self.assertEqual(changed["tasks"]["producer"]["status"], "implemented")
        with self.assertRaisesRegex(ValueError, "not user-confirmed"):
            self.check("producer")

    def test_missing_gpu_facts_require_disclosed_issue_acknowledgment(self):
        self.report["identity"]["gpu"]["models"] = []
        self.report["readiness"]["issues"] = ["MetaX GPU models could not be collected"]
        self.call("refresh")
        with self.assertRaisesRegex(ValueError, "acknowledge"):
            self.confirm()
        self.confirm(acknowledge=True)
        self.assertTrue(
            self.read()["environment"]["confirmation"]["acknowledged_issues"]
        )

    def test_runtime_origin_from_another_wheel_is_rejected(self):
        self.confirm()
        self.task("producer", requires=("closure",))
        wheel = self.root / "other-wheel.py"
        wheel.write_text("# installed wheel stand-in\n")
        report = {
            "vllm": str(wheel),
            "vllm_metax": str(wheel),
            "dispatch": {"backend": "fixture"},
        }
        code = (
            "import json, os; from pathlib import Path; "
            "Path(os.environ['METAX_UPGRADE_ORIGINS_FILE']).write_text("
            f"json.dumps({report!r}))"
        )
        self.assertEqual(self.check("producer", code, "closure"), 1)
        latest = self.read()["tasks"]["producer"]["evidence"][-1]
        self.assertIn("does not match", latest["origins"])

    def test_hard_quota_stop_leaves_interrupted_check_and_resume_todo(self):
        self.confirm()
        self.task("producer")
        self.call("task-set", "--id", "producer", "--status", "in_progress")
        report_file = self.root / "mock-environment.json"
        report_file.write_text(json.dumps(self.report))
        pid_file = self.root / "check.pid"
        child_code = (
            "import os, time; from pathlib import Path; "
            f"Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(30)"
        )
        arguments = [
            "run-check",
            "--run-dir",
            str(self.run),
            "--id",
            "producer",
            "--kind",
            "static",
            "--configuration",
            "test-config",
            "--timeout",
            "60",
            "--",
            sys.executable,
            "-c",
            child_code,
        ]
        launcher = (
            "import json; from pathlib import Path; from unittest.mock import patch; "
            "from metax_upgrade import upgrade; "
            f"report=json.loads(Path({str(report_file)!r}).read_text()); "
            "upgrade.collect=lambda *_a, **_k: report; "
            f"upgrade.main({arguments!r})"
        )
        process = subprocess.Popen(
            [sys.executable, "-c", launcher],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            deadline = time.monotonic() + 5
            while not pid_file.is_file() and time.monotonic() < deadline:
                if process.poll() is not None:
                    self.fail(process.stderr.read().decode())
                time.sleep(0.02)
            self.assertTrue(pid_file.is_file(), "Validation child failed to start")
            process.kill()
            process.wait(timeout=5)
            child_pid = int(pid_file.read_text())
            deadline = time.monotonic() + 5
            child_running = True
            while time.monotonic() < deadline:
                try:
                    # A killed child can remain an init-owned zombie in a container.
                    child_stat = Path(f"/proc/{child_pid}/stat").read_text()
                    child_running = child_stat.rsplit(")", 1)[1].split()[0] != "Z"
                except FileNotFoundError:
                    child_running = False
                if not child_running:
                    break
                time.sleep(0.02)
            self.assertFalse(
                child_running, "Quota interruption left the validation child running"
            )
            persisted = self.read()
            self.assertEqual(
                persisted["tasks"]["producer"]["evidence"][-1]["result"], "interrupted"
            )
            self.call("resume")
            recovered = self.read()["tasks"]["producer"]
            self.assertEqual(recovered["status"], "todo")
            self.assertEqual(
                state.missing_evidence(self.read(), "producer"), ["static"]
            )
            self.assertIn("Interrupted work", (self.run / "TODO.md").read_text())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            process.stderr.close()
            if pid_file.is_file():
                try:
                    os.killpg(int(pid_file.read_text()), signal.SIGKILL)
                except ProcessLookupError:
                    pass


if __name__ == "__main__":
    unittest.main()
