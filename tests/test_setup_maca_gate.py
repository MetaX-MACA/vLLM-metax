import ast
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from packaging.version import Version, parse


def load_setup_helpers(use_maca: bool):
    setup_path = Path(__file__).parents[1] / "setup.py"
    module = ast.parse(setup_path.read_text(encoding="utf-8"), filename=str(setup_path))
    selected = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_is_maca", "get_maca_version"}
    ]

    namespace = {
        "os": os,
        "parse": parse,
        "Version": Version,
        "USE_MACA": use_maca,
        "VLLM_TARGET_DEVICE": "cuda",
        "torch": type("TorchStub", (), {"version": type("VersionStub", (), {"cuda": "12.8"})()})(),
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(setup_path), "exec"),
        namespace,
    )
    return namespace["_is_maca"], namespace["get_maca_version"]


class SetupMACAGateTest(unittest.TestCase):
    def test_is_maca_requires_use_maca_flag(self):
        is_maca, _ = load_setup_helpers(use_maca=False)
        self.assertFalse(is_maca())

    def test_get_maca_version_allows_missing_path_when_use_maca_disabled(self):
        _, get_maca_version = load_setup_helpers(use_maca=False)
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(get_maca_version())


if __name__ == "__main__":
    unittest.main()
