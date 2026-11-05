# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Relocated entry points must work independently of the caller's directory."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from metax_upgrade import environment, lifecycle


class LayoutTests(unittest.TestCase):
    def test_default_checkout_and_rule_identity_use_project_root(self):
        root = Path(__file__).resolve().parents[6]
        self.assertEqual(environment._discover_checkout("vllm_metax"), root)
        files = lifecycle.runtime_identity()["files"]
        for path in (
            "AGENTS.md",
            ".agents/agents/vllm-metax-upgrade.md",
            ".agents/skills/common-sense/references/tooling.md",
            ".agents/skills/common-sense/scripts/metax_upgrade/upgrade.py",
        ):
            self.assertIn(path, files)

    def test_cli_and_probe_find_bundled_helpers_from_another_directory(self):
        package = Path(environment.__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            for script in (
                package / "upgrade.py",
                package.parent / "probe_environment.py",
            ):
                result = subprocess.run(
                    [sys.executable, "-B", str(script), "--help"],
                    cwd=directory,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout)

            # Substitute only the expensive environment inventory, retaining the
            # real probe wrapper and its relative helper lookup and child command.
            probe = Path(directory) / "scripts/probe_environment.py"
            probe.parent.mkdir()
            probe.write_bytes((package.parent / "probe_environment.py").read_bytes())
            helper = probe.parent / "metax_upgrade/environment.py"
            helper.parent.mkdir()
            helper.write_text("import json, sys\nprint(json.dumps(sys.argv[1:]))\n")
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(probe),
                    "--vllm-source",
                    directory,
                    "--metax-source",
                    directory,
                ],
                cwd=directory,
                capture_output=True,
                text=True,
                timeout=15,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(
                json.loads(result.stdout),
                [
                    "--source-probe",
                    "--vllm-source",
                    directory,
                    "--metax-source",
                    directory,
                    "--runtime-cwd",
                    directory,
                ],
            )
