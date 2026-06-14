import ast
import os
from pathlib import Path
from unittest.mock import patch


def _load_bool_parser():
    source_path = Path(__file__).resolve().parents[1] / "vllm_metax" / "envs.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Import)
            and any(alias.name == "os" for alias in node.names)
        )
        or (isinstance(node, ast.FunctionDef) and node.name == "_parse_bool_env")
    ]
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["_parse_bool_env"]


def test_bool_parser_accepts_zero_and_one():
    parse_bool = _load_bool_parser()
    with patch.dict(os.environ, {"FLAG": "1"}, clear=True):
        assert parse_bool("FLAG", "0") is True
    with patch.dict(os.environ, {"FLAG": "0"}, clear=True):
        assert parse_bool("FLAG", "1") is False


def test_bool_parser_reports_variable_name():
    parse_bool = _load_bool_parser()
    with patch.dict(os.environ, {"FLAG": "yes"}, clear=True):
        try:
            parse_bool("FLAG", "0")
        except ValueError as err:
            assert "FLAG" in str(err)
            assert "0 or 1" in str(err)
        else:
            raise AssertionError("expected ValueError")


if __name__ == "__main__":
    test_bool_parser_accepts_zero_and_one()
    test_bool_parser_reports_variable_name()
