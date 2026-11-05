# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Audit local model dependencies without importing the model or plugin."""

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path


PUBLIC = (
    "vllm.model_executor.models.utils",
    "vllm.model_executor.models.interfaces",
    "vllm.model_executor.models.registry",
    "vllm.model_executor.models.config",
    "vllm.model_executor.models.adapters",
    "vllm.models.common",
)


def concrete(module):
    if any(module == prefix or module.startswith(prefix + ".") for prefix in PUBLIC):
        return False
    return module.startswith(("vllm.models.", "vllm.model_executor.models."))


def audit(root, entries):
    root = Path(root).resolve()
    pending = [root / entry for entry in entries]
    seen = {}
    violations = []
    dynamic = []
    boundaries = set()
    while pending:
        path = pending.pop().resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(
                "Model dependency must be a local source file: " + str(path)
            )
        relative = str(path.relative_to(root))
        if relative in seen:
            continue
        seen[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        module = ".".join(path.relative_to(root).with_suffix("").parts)
        package = (
            module.rsplit(".", 1)[0]
            if path.name != "__init__.py"
            else module.rsplit(".", 1)[0]
        )
        tree = ast.parse(path.read_text(), filename=str(path))
        imports = []
        aliases = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for item in node.names:
                    if item.asname:
                        aliases[item.asname] = item.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                for item in node.names:
                    aliases[item.asname or item.name] = node.module + "." + item.name
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend((item.name, node.lineno) for item in node.names)
            elif isinstance(node, ast.ImportFrom):
                target = node.module or ""
                if node.level:
                    target = importlib.util.resolve_name(
                        "." * node.level + target, package
                    )
                imports.append((target, node.lineno))
                # "from vllm.models import deepseek_v4" is a concrete import too.
                if target in {"vllm.models", "vllm.model_executor.models"}:
                    for alias in node.names:
                        if alias.name not in {
                            "ModelRegistry",
                            "ModelRegistryEntry",
                            "interfaces",
                            "utils",
                            "registry",
                            "config",
                            "adapters",
                            "common",
                        }:
                            imports.append((target + "." + alias.name, node.lineno))
                imports.extend(
                    (target + "." + item.name, node.lineno)
                    for item in node.names
                    if target.startswith("vllm_metax")
                )
            elif (
                isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and aliases.get(node.value.id)
                in {"vllm.models", "vllm.model_executor.models"}
            ):
                if node.attr not in {
                    "ModelRegistry",
                    "ModelRegistryEntry",
                    "interfaces",
                    "utils",
                    "registry",
                    "config",
                    "adapters",
                    "common",
                }:
                    imports.append(
                        (aliases[node.value.id] + "." + node.attr, node.lineno)
                    )
            elif isinstance(node, ast.Call) and (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "import_module"
                or isinstance(node.func, ast.Name)
                and (
                    node.func.id == "__import__"
                    or aliases.get(node.func.id) == "importlib.import_module"
                )
            ):
                if (
                    node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                ):
                    imports.append((node.args[0].value, node.lineno))
                else:
                    dynamic.append(
                        {
                            "file": relative,
                            "line": node.lineno,
                            "reason": "Nonliteral dynamic import requires explicit review",
                        }
                    )
        for target, line in imports:
            if concrete(target):
                violations.append({"file": relative, "line": line, "module": target})
            elif target.startswith("vllm_metax"):
                candidate = root.joinpath(*target.split("."))
                if candidate.with_suffix(".py").is_file():
                    pending.append(candidate.with_suffix(".py"))
                elif (candidate / "__init__.py").is_file():
                    pending.append(candidate / "__init__.py")
            elif target.startswith("vllm"):
                boundaries.add(target)
    return {
        "files": seen,
        "prohibited_imports": violations,
        "dynamic_imports": dynamic,
        "public_framework_boundaries": sorted(boundaries),
        "scope": "Local implementation dependencies; upstream public framework internals and runtime dispatch require final worker review",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--entry", action="append", required=True)
    args = parser.parse_args()
    report = audit(args.source, args.entry)
    print(json.dumps(report, indent=2))
    raise SystemExit(
        1 if report["prohibited_imports"] or report["dynamic_imports"] else 0
    )
