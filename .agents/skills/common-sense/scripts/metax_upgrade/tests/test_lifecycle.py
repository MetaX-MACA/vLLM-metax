# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Crash recovery and evidence identity checks without importing target packages."""

import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from metax_upgrade import lifecycle, state as store


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.metax = self.root / "metax"
        self.run = self.root / "run"
        self.source = self.metax / "vllm_metax/op.py"
        self.source.parent.mkdir(parents=True)
        self.source.write_text("original = 1\n")
        self.state = {
            "schema_version": store.SCHEMA_VERSION,
            "revision": 0,
            "inputs": {
                "metax_source": str(self.metax),
                "scope": "test",
                "mode": "component",
            },
            "tasks": {},
            "contracts": {},
            "checkpoints": [],
            "environment": {"fingerprint": "pending", "snapshot": "pending.json"},
            "initialization": {"phase": "baseline"},
        }
        self.state["source_manifest"] = store.source_manifest(self.state)
        store.save(self.run, self.state)
        self.archive = self.run / "baseline/metax-source.zip"

    def test_mid_archive_interruption_is_resumable_and_partial_is_preserved(self):
        original_write = zipfile.ZipFile.write

        def interrupt(archive, *args, **kwargs):
            original_write(archive, *args, **kwargs)
            raise InterruptedError("injected interruption")

        with patch.object(zipfile.ZipFile, "write", interrupt):
            with self.assertRaises(InterruptedError):
                lifecycle.prepare_baseline(self.run, self.state)
        self.assertEqual(store.load(self.run)["initialization"]["phase"], "baseline")
        partials = list(self.archive.parent.glob("*.partial"))
        self.assertEqual(len(partials), 1)
        self.assertFalse(self.archive.exists())
        lifecycle.prepare_baseline(self.run, store.load(self.run))
        self.assertTrue(partials[0].exists())
        self.assertEqual(store.load(self.run)["initialization"]["phase"], "environment")
        with zipfile.ZipFile(self.archive) as archive:
            self.assertEqual(archive.read("vllm_metax/op.py"), b"original = 1\n")

    def test_direct_extensionless_and_path_executables_are_hashed(self):
        script = self.root / "validator"
        script.write_text("#!/bin/sh\nexit 0\n")
        script.chmod(0o755)
        first = lifecycle.evidence_inputs([str(script)], self.root)
        self.assertEqual(first[str(script)], store.file_hash(script))
        with patch.dict(lifecycle.os.environ, {"PATH": str(self.root)}):
            self.assertEqual(first, lifecycle.evidence_inputs(["validator"], self.root))
        script.write_text("#!/bin/sh\nexit 1\n")
        self.assertNotEqual(first, lifecycle.evidence_inputs([str(script)], self.root))

    def test_hard_killed_capture_recovers_from_persisted_initial_state(self):
        code = """
import os, signal, sys, zipfile
from metax_upgrade import lifecycle, state as store
original_write = zipfile.ZipFile.write
def killed(archive, *args, **kwargs):
    original_write(archive, *args, **kwargs)
    os.kill(os.getpid(), signal.SIGKILL)
zipfile.ZipFile.write = killed
lifecycle.prepare_baseline(sys.argv[1], store.load(sys.argv[1]))
"""
        process = subprocess.run(
            [sys.executable, "-c", code, str(self.run)],
            capture_output=True,
            timeout=10,
        )
        self.assertEqual(process.returncode, -signal.SIGKILL, process.stderr.decode())
        self.assertFalse(self.archive.exists())
        self.assertTrue(list(self.archive.parent.glob("*.partial")))
        lifecycle.prepare_baseline(self.run, store.load(self.run))
        self.assertTrue(
            lifecycle._archive_valid(self.archive, self.state["source_manifest"])
        )

    def test_final_archive_is_atomic_and_valid_before_commit(self):
        original_replace = lifecycle.os.replace
        checked = []

        def check_replace(source, target):
            if Path(target) == self.archive:
                self.assertFalse(self.archive.exists())
                self.assertTrue(
                    lifecycle._archive_valid(source, self.state["source_manifest"])
                )
                checked.append(True)
            return original_replace(source, target)

        with patch.object(lifecycle.os, "replace", check_replace):
            lifecycle.prepare_baseline(self.run, self.state)
        self.assertEqual(checked, [True])

    def test_committed_archive_survives_crash_before_state_save_and_source_drift(self):
        with patch.object(store, "save", side_effect=InterruptedError("before save")):
            with self.assertRaises(InterruptedError):
                lifecycle.prepare_baseline(self.run, self.state)
        original = self.archive.read_bytes()
        self.source.write_text("changed_after_committed_baseline = 2\n")
        lifecycle.prepare_baseline(self.run, store.load(self.run))
        self.assertEqual(self.archive.read_bytes(), original)
        self.assertEqual(store.load(self.run)["initialization"]["phase"], "environment")

    def test_incomplete_original_capture_refuses_changed_or_new_sources(self):
        self.archive.parent.mkdir(parents=True)
        self.archive.write_bytes(b"partial old archive")
        (self.source.parent / "untracked.py").write_text("new file\n")
        with self.assertRaisesRegex(ValueError, "Source changed.*incomplete baseline"):
            lifecycle.prepare_baseline(self.run, self.state)
        self.assertEqual(self.archive.read_bytes(), b"partial old archive")

    def test_invalid_original_archive_is_preserved_before_safe_rebuild(self):
        self.archive.parent.mkdir(parents=True)
        self.archive.write_bytes(b"old failed initialization")
        lifecycle.prepare_baseline(self.run, self.state)
        interrupted = list(self.archive.parent.glob("metax-source.interrupted-*.zip"))
        self.assertEqual(len(interrupted), 1)
        self.assertEqual(interrupted[0].read_bytes(), b"old failed initialization")
        self.assertTrue(
            lifecycle._archive_valid(self.archive, self.state["source_manifest"])
        )

    def test_source_change_during_capture_cannot_commit(self):
        original_write = zipfile.ZipFile.write

        def change_source(archive, *args, **kwargs):
            self.source.write_text("changed_during_capture = 3\n")
            original_write(archive, *args, **kwargs)

        with patch.object(zipfile.ZipFile, "write", change_source):
            with self.assertRaisesRegex(ValueError, "Source changed while capturing"):
                lifecycle.prepare_baseline(self.run, self.state)
        self.assertFalse(self.archive.exists())
        self.assertTrue(list(self.archive.parent.glob("*.partial")))

    def test_existing_manifest_must_match_frozen_state(self):
        manifest = self.run / "baseline/source-manifest.json"
        store.write_json(manifest, {"vllm_metax/op.py": "0" * 64})
        with self.assertRaisesRegex(ValueError, "disagrees with saved initialization"):
            lifecycle.prepare_baseline(self.run, self.state)
        self.assertEqual(json.loads(manifest.read_text())["vllm_metax/op.py"], "0" * 64)

    def test_runtime_identity_tracks_rules_and_tool_changes_but_not_runs(self):
        tool = (
            self.root
            / ".agents/skills/common-sense/scripts/metax_upgrade/upgrade.py"
        )
        tool.parent.mkdir(parents=True)
        tool.write_text("tool = 1\n")
        rule = self.root / ".agents/skills/common-sense/SKILL.md"
        rule.parent.mkdir(parents=True, exist_ok=True)
        rule.write_text("initial rules\n")
        original = lifecycle.runtime_identity(self.root)
        rule.write_text("changed rules\n")
        changed_rule = lifecycle.runtime_identity(self.root)
        self.assertNotEqual(original["fingerprint"], changed_rule["fingerprint"])
        tool.write_text("tool = 2\n")
        changed_tool = lifecycle.runtime_identity(self.root)
        self.assertNotEqual(changed_rule["fingerprint"], changed_tool["fingerprint"])
        agent = self.root / ".agents/agents/vllm-metax-upgrade.md"
        agent.parent.mkdir(parents=True)
        agent.write_text("coordinator instructions\n")
        changed_agent = lifecycle.runtime_identity(self.root)
        self.assertNotEqual(changed_tool["fingerprint"], changed_agent["fingerprint"])
        entry = self.root / "AGENTS.md"
        entry.write_text("workflow discovery instructions\n")
        changed_entry = lifecycle.runtime_identity(self.root)
        self.assertNotEqual(changed_agent["fingerprint"], changed_entry["fingerprint"])
        ignored = self.root / ".metax-upgrade/run/environment.json"
        ignored.parent.mkdir(parents=True)
        ignored.write_text("historical evidence\n")
        self.assertEqual(changed_entry, lifecycle.runtime_identity(self.root))

    def test_evidence_inputs_track_script_data_and_inline_without_interpreter(self):
        script = self.root / "verify.py"
        script.write_text("print('test')\n")
        data = self.root / "inputs.json"
        data.write_text("{}\n")
        identity = lifecycle.evidence_inputs(
            [sys.executable, str(script), "--data=inputs.json"], self.root
        )
        self.assertEqual(set(identity), {str(script), str(data)})
        data.write_text('{"value": 1}\n')
        self.assertNotEqual(
            identity,
            lifecycle.evidence_inputs(
                [sys.executable, str(script), "--data=inputs.json"], self.root
            ),
        )
        first = lifecycle.evidence_inputs([sys.executable, "-c", "print(1)"], self.root)
        second = lifecycle.evidence_inputs(
            [sys.executable, "-c", "print(2)"], self.root
        )
        self.assertEqual(list(first), ["<python -c>"])
        self.assertNotEqual(first, second)
        with self.assertRaisesRegex(ValueError, "not an existing file"):
            lifecycle.evidence_inputs(
                [sys.executable, "-c", "pass"], self.root, ["missing"]
            )

    def test_python_module_resolution_never_imports_package(self):
        package = self.root / "checkmod"
        package.mkdir()
        initializer = package / "__init__.py"
        initializer.write_text("raise RuntimeError('must never import this')\n")
        main = package / "__main__.py"
        main.write_text("print('check')\n")
        identity = lifecycle.evidence_inputs(
            [sys.executable, "-m", "checkmod"], self.root
        )
        self.assertEqual(set(identity), {str(initializer), str(main)})


if __name__ == "__main__":
    unittest.main()
