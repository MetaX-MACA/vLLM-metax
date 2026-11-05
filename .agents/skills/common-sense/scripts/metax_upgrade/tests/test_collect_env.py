# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Offline regression checks for the collector in pip-free uv environments."""

import contextlib
import importlib.util
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
import venv
from unittest.mock import Mock, patch


COLLECTOR_PATH = Path(__file__).resolve().parents[6] / "vllm_metax/collect_env.py"


def load_collector():
    # NOTE(metax): Load the entry file without executing plugin initialization.
    spec = importlib.util.spec_from_file_location(
        "metax_collect_env_test", COLLECTOR_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PipFreeEnvironmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.collector = load_collector()

    def test_pip_preferred_and_lists_all_packages_for_selected_python(self):
        calls = []

        def run(command):
            calls.append(command)
            return 0, "torch==2.10.0+metax\nunrelated==1", ""

        with patch("importlib.util.find_spec", return_value=object()):
            label, packages = self.collector.get_pip_packages(run)
        self.assertIn("pip", label.lower())
        self.assertEqual(calls[0][0], sys.executable)
        self.assertIn("torch==2.10.0+metax", packages)
        self.assertIn("unrelated==1", packages)
        self.assertEqual(len(calls), 1)

    def test_uv_works_without_uv_environment_variable(self):
        calls = []

        def run(command):
            calls.append(command)
            return 0, "torch==2.10.0+metax\nunrelated==1", ""

        stdout = io.StringIO()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("importlib.util.find_spec", return_value=None),
            patch.object(self.collector.shutil, "which", return_value="/tools/uv"),
            contextlib.redirect_stdout(stdout),
        ):
            label, packages = self.collector.get_pip_packages(run, {"torch"})
        self.assertIn("uv", label.lower())
        self.assertEqual(calls[0][0], "/tools/uv")
        self.assertEqual(calls[0][1:3], ["pip", "list"])
        self.assertIn("--python", calls[0])
        self.assertEqual(calls[0][calls[0].index("--python") + 1], sys.executable)
        self.assertIn("torch==2.10.0+metax", packages)
        self.assertNotIn("unrelated", packages)
        self.assertEqual(stdout.getvalue(), "")

    def test_failed_pip_falls_back_to_uv(self):
        calls = []

        def run(command):
            calls.append(command)
            if command[0] == sys.executable:
                return 1, "", "No module named pip"
            return 0, "torch==2.10.0", ""

        with (
            patch("importlib.util.find_spec", return_value=object()),
            patch.object(self.collector.shutil, "which", return_value="/tools/uv"),
        ):
            label, packages = self.collector.get_pip_packages(run, {"torch"})
        self.assertIn("uv", label.lower())
        self.assertEqual(packages, "torch==2.10.0")
        self.assertEqual(len(calls), 2)

    def test_failed_package_commands_use_installed_metadata(self):
        class Distribution:
            metadata = {"Name": "torch"}
            version = "2.10.0+metax"

        with (
            patch("importlib.util.find_spec", return_value=None),
            patch.object(self.collector.shutil, "which", return_value="/tools/uv"),
            patch.object(
                self.collector.metadata,
                "distributions",
                return_value=[
                    Distribution(),
                    SimpleNamespace(metadata={"Name": "auxiliary"}, version="7"),
                ],
            ),
        ):
            label, packages = self.collector.get_pip_packages(
                lambda command: (1, "", "uv failed")
            )
        self.assertIn("metadata", label.lower())
        self.assertEqual(packages.splitlines(), ["auxiliary==7", "torch==2.10.0+metax"])

    def test_no_package_manager_and_no_distributions_is_valid(self):
        with (
            patch("importlib.util.find_spec", return_value=None),
            patch.object(self.collector.shutil, "which", return_value=None),
            patch.object(self.collector.metadata, "distributions", return_value=[]),
        ):
            label, packages = self.collector.get_pip_packages(
                lambda command: self.fail("No package manager should be executed"),
                {"torch"},
            )
        self.assertIn("metadata", label.lower())
        self.assertEqual(packages, "")


class MissingRuntimeTests(unittest.TestCase):
    def test_legacy_collection_handles_missing_torch(self):
        collector = load_collector()
        with (
            patch.object(collector, "_load_runtime_packages"),
            patch.object(collector, "TORCH_AVAILABLE", False),
            patch.object(collector, "VLLM_AVAILABLE", False),
            patch.object(collector, "VLLM_METAX_AVAILABLE", False),
            patch.object(collector, "run", return_value=(127, "", "unavailable")),
            patch.object(
                collector, "get_pip_packages", return_value=("importlib.metadata", "")
            ),
            patch.object(collector, "get_env_vars", return_value=""),
        ):
            report = collector.get_env_info()
            pretty = collector.pretty_str(report)
        self.assertIn("PyTorch", pretty)
        self.assertIn(report.torch_version, (None, "N/A"))
        self.assertIn(report.is_debug_build, (None, "N/A"))
        self.assertIn(report.is_cuda_available, (None, "N/A"))
        self.assertIn(report.cuda_compiled_version, (None, "N/A"))

    def test_cuda_initialization_failure_keeps_diagnostic_collection_alive(self):
        collector = load_collector()
        cuda = Mock()
        cuda.is_available.return_value = True
        cuda.init.side_effect = RuntimeError("driver startup failed")
        with (
            patch.object(collector, "TORCH_AVAILABLE", True),
            patch.object(collector, "torch", SimpleNamespace(cuda=cuda), create=True),
            patch.dict(os.environ, {"CUDA_MODULE_LOADING": "LAZY"}),
        ):
            value = collector.get_cuda_module_loading_config()
        cuda.init.assert_called_once()
        self.assertIsInstance(value, str)
        self.assertTrue(value)
        self.assertNotEqual(value, "LAZY")

    def test_existing_pretty_and_main_flow_renders_collected_report(self):
        collector = load_collector()
        with (
            patch.object(collector, "_load_runtime_packages"),
            patch.object(collector, "run", return_value=(127, "", "unavailable")),
            patch.object(
                collector,
                "get_pip_packages",
                return_value=("importlib.metadata", "auxiliary==7"),
            ),
        ):
            report = collector.get_env_info()
        expected = collector.pretty_str(report)
        with (
            patch.object(collector, "get_env_info", return_value=report) as get_info,
            patch.object(collector, "pretty_str", wraps=collector.pretty_str) as render,
        ):
            self.assertEqual(collector.get_pretty_env_info(), expected)
        get_info.assert_called_once_with()
        render.assert_called_once_with(report)
        output = io.StringIO()
        with (
            patch.object(
                collector, "get_pretty_env_info", return_value=expected
            ) as get_pretty,
            contextlib.redirect_stdout(output),
        ):
            collector.main()
        get_pretty.assert_called_once_with()
        self.assertIn(expected, output.getvalue())
        self.assertIn("auxiliary==7", output.getvalue())


class CollectorImportTests(unittest.TestCase):
    def test_entry_import_needs_only_stdlib_and_does_not_activate_plugin(self):
        source = r"""
import importlib.abc
import importlib.util
import sys

class RejectRuntimeImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'vllm', 'vllm_metax', 'regex'}:
            raise RuntimeError('Runtime import attempted: ' + fullname)

sys.meta_path.insert(0, RejectRuntimeImports())
spec = importlib.util.spec_from_file_location('isolated_metax_collector', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(module.__name__)
"""
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-c", source, str(COLLECTOR_PATH)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "isolated_metax_collector")


class CollectorCLITests(unittest.TestCase):
    def test_plain_standalone_script_in_pip_free_venv_can_redirect_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            isolated = root / "single-file"
            isolated.mkdir()
            script = isolated / "collect_env.py"
            script.write_bytes(COLLECTOR_PATH.read_bytes())
            environment = root / "pip-free"
            venv.EnvBuilder(with_pip=False).create(environment)
            python = environment / (
                "Scripts/python.exe" if os.name == "nt" else "bin/python"
            )
            child_env = os.environ.copy()
            child_env.pop("PYTHONPATH", None)
            child_env["VIRTUAL_ENV"] = str(environment)
            child_env["UV_OFFLINE"] = "1"
            child_env["UV_CACHE_DIR"] = str(root / "uv-cache")
            destination = root / "environment.txt"
            with destination.open("w") as output:
                result = subprocess.run(
                    [str(python), str(script)],
                    env=child_env,
                    cwd=root,
                    stdout=output,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=45,
                )
            self.assertEqual(result.returncode, 0, result.stderr)
            text = destination.read_text()
            for section in (
                "System Info",
                "Python Environment",
                "PyTorch Info",
                "Versions of",
            ):
                self.assertIn(section, text)
            self.assertIn(str(python), text)
            self.assertIn("=" * 30, text)
            self.assertNotIn("Traceback", text + result.stderr)
            self.assertNotIn("\x1b[", text)
            self.assertEqual(
                {entry.name for entry in isolated.iterdir()}, {"collect_env.py"}
            )


if __name__ == "__main__":
    unittest.main()
