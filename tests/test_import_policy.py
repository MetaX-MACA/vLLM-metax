# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import ast
from pathlib import Path

import pytest


TEST_ROOT = Path(__file__).parent
NON_TEST_MODULES = {
    "__init__.py",
    "conftest.py",
    "test_import_policy.py",
}


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


@pytest.mark.parametrize(
    "path",
    sorted(
        path
        for path in TEST_ROOT.rglob("*.py")
        if path.name not in NON_TEST_MODULES
    ),
    ids=lambda path: str(path.relative_to(TEST_ROOT)),
)
def test_test_modules_import_plugin_subject(path: Path):
    modules = _imported_modules(path)
    imports_plugin = any(
        module == "vllm_metax" or module.startswith("vllm_metax.")
        for module in modules
    )
    assert imports_plugin, (
        f"{path.relative_to(TEST_ROOT)} does not import vllm_metax; "
        "upstream-only tests do not belong in the plugin suite"
    )


@pytest.mark.parametrize(
    "path",
    sorted(
        path
        for path in TEST_ROOT.rglob("*.py")
        if path.name not in {"__init__.py", "test_import_policy.py"}
    ),
    ids=lambda path: str(path.relative_to(TEST_ROOT)),
)
def test_vllm_imports_are_not_the_only_subject(path: Path):
    modules = _imported_modules(path)
    imports_upstream = any(
        module == "vllm" or module.startswith("vllm.") for module in modules
    )
    imports_plugin = any(
        module == "vllm_metax" or module.startswith("vllm_metax.")
        for module in modules
    )
    if imports_upstream:
        assert imports_plugin, (
            f"{path.relative_to(TEST_ROOT)} imports upstream vLLM without also "
            "importing the plugin subject"
        )
