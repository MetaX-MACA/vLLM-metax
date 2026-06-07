import ast
import json
from collections import namedtuple
from pathlib import Path


def _load_json_helper():
    source_path = Path(__file__).resolve().parents[1] / "vllm_metax" / "collect_env.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    nodes = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Import)
            and any(alias.name == "json" for alias in node.names)
        )
        or (isinstance(node, ast.FunctionDef) and node.name == "get_json_env_info")
    ]
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace


def test_collect_env_json_uses_namedtuple_fields():
    namespace = _load_json_helper()
    env_type = namedtuple("SystemEnv", ["maca_sdk_version", "torch_version"])
    namespace["get_env_info"] = lambda: env_type("2.33.1", "2.8.0")

    data = json.loads(namespace["get_json_env_info"]())

    assert data == {"maca_sdk_version": "2.33.1", "torch_version": "2.8.0"}


if __name__ == "__main__":
    test_collect_env_json_uses_namedtuple_fields()
