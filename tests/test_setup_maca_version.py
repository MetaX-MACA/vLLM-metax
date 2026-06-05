import ast
import os
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from packaging.version import Version, parse


def load_setup_version_helpers():
    setup_path = Path(__file__).parents[1] / "setup.py"
    module = ast.parse(setup_path.read_text(encoding="utf-8"), filename=str(setup_path))
    selected = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name
        in {
            "get_maca_version",
            "fixed_version_scheme",
            "always_hash",
            "get_plugin_version",
        }
    ]

    helpers = types.SimpleNamespace()
    namespace = {
        "os": os,
        "parse": parse,
        "Version": Version,
        "ScmVersion": object,
        "get_version": lambda **_kwargs: "0.20.0+abc1234.d20260101",
        "torch": types.SimpleNamespace(__version__="2.8.0+metax"),
        "_is_maca": lambda: True,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(setup_path), "exec"), namespace)

    helpers.get_maca_version = namespace["get_maca_version"]
    helpers.get_plugin_version = namespace["get_plugin_version"]
    return helpers


class SetupMACAVersionTest(unittest.TestCase):
    def setUp(self):
        self.helpers = load_setup_version_helpers()

    def test_get_maca_version_reads_maca_home_when_maca_path_is_unset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "Version.txt").write_text("MACA Version: 2.32.0\n", encoding="utf-8")
            with patch.dict(os.environ, {"MACA_HOME": tmpdir}, clear=True):
                self.assertEqual(str(self.helpers.get_maca_version()), "2.32.0")

    def test_get_maca_version_returns_zero_when_path_is_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(str(self.helpers.get_maca_version()), "0")

    def test_get_plugin_version_marks_missing_maca_version_as_unknown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {"MACA_PATH": tmpdir}, clear=True):
                version = self.helpers.get_plugin_version()

        self.assertIn("macaunknown", version)
        self.assertIn("torch2.8", version)


if __name__ == "__main__":
    unittest.main()
