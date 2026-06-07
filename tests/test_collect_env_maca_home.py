import ast
import os
from pathlib import Path
from unittest.mock import patch


def _load_collect_env_helpers():
    source_path = Path(__file__).resolve().parents[1] / "vllm_metax" / "collect_env.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    wanted = {"get_maca_home", "get_maca_sdk_version", "run_and_parse_first_match"}
    nodes = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom))
        and (not isinstance(node, ast.FunctionDef) or node.name in wanted)
    ]
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"TORCH_AVAILABLE": False}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace


def test_collect_env_reads_maca_home_when_maca_path_is_missing(tmp_path):
    helpers = _load_collect_env_helpers()
    (tmp_path / "Version.txt").write_text("MACA Version: 2.33.1\n", encoding="utf-8")

    def fake_run(command):
        assert command == ["cat", os.path.join(str(tmp_path), "Version.txt")]
        return 0, "MACA Version: 2.33.1", ""

    with patch.dict(os.environ, {"MACA_HOME": str(tmp_path)}, clear=True):
        assert helpers["get_maca_sdk_version"](fake_run).strip() == "2.33.1"


def test_collect_env_returns_none_without_maca_sdk_path():
    helpers = _load_collect_env_helpers()

    with patch.dict(os.environ, {}, clear=True):
        assert helpers["get_maca_sdk_version"](lambda command: (1, "", "")) is None
