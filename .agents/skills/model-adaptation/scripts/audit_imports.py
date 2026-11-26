# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Audit local model dependencies without importing the model or plugin."""

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import regex as re
import subprocess


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


def _audit(root, entries, declared):
    pending = [root / entry for entry in entries]
    seen = {}
    violations = []
    dynamic = []
    boundaries = set()
    reused = {}
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
                imports.extend((item.name, node.lineno, None) for item in node.names)
            elif isinstance(node, ast.ImportFrom):
                target = node.module or ""
                if node.level:
                    target = importlib.util.resolve_name(
                        "." * node.level + target, package
                    )
                if concrete(target):
                    imports.extend(
                        (target, node.lineno, item.name) for item in node.names
                    )
                else:
                    imports.append((target, node.lineno, None))
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
                            imports.append(
                                (target + "." + alias.name, node.lineno, None)
                            )
                imports.extend(
                    (target + "." + item.name, node.lineno, None)
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
                        (aliases[node.value.id] + "." + node.attr, node.lineno, None)
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
                    imports.append((node.args[0].value, node.lineno, None))
                else:
                    dynamic.append(
                        {
                            "file": relative,
                            "line": node.lineno,
                            "reason": "Nonliteral dynamic import requires explicit review",
                        }
                    )
        for target, line, name in imports:
            if concrete(target):
                key = (relative, f"{target}.{name}")
                if name is not None and key in declared:
                    reused.setdefault(key, []).append(line)
                else:
                    violation = {"file": relative, "line": line, "module": target}
                    if name is not None:
                        violation["symbol"] = name
                    violations.append(violation)
            elif target.startswith("vllm_metax"):
                candidate = root.joinpath(*target.split("."))
                if candidate.with_suffix(".py").is_file():
                    pending.append(candidate.with_suffix(".py"))
                elif (candidate / "__init__.py").is_file():
                    pending.append(candidate / "__init__.py")
            elif target.startswith("vllm"):
                boundaries.add(target)
    if unused := declared.keys() - reused.keys():
        raise ValueError(
            f"Declared component imports were not reached: {sorted(unused)}"
        )
    return {
        "files": seen,
        "prohibited_imports": violations,
        "dynamic_imports": dynamic,
        "public_framework_boundaries": sorted(boundaries),
        "upstream_component_reuse": [
            {**declared[key], "lines": lines} for key, lines in sorted(reused.items())
        ],
        "scope": "Local implementation dependencies; upstream public framework internals and runtime dispatch require final worker review",
    }


def _declarations(root, entries, specs, upstream_source):
    declarations = {}
    for spec in specs:
        path = (root / spec).resolve()
        if not path.is_relative_to(root / "vllm_metax/models"):
            raise ValueError("Reuse declarations belong to a local model spec")
        text = path.read_text()
        headings = re.findall(r"^## Upstream component reuse[ \t]*$", text, re.M)
        if not headings:
            continue
        blocks = re.findall(
            r"^## Upstream component reuse[ \t]*\n\s*```json[ \t]*\n(.*?)^```[ \t]*$",
            text,
            re.M | re.S,
        )
        if len(headings) != 1 or len(blocks) != 1:
            raise ValueError(f"Expected one upstream reuse JSON block in {spec}")
        rows = json.loads(blocks[0])
        if not isinstance(rows, list) or not rows:
            raise ValueError(
                f"Upstream component reuse needs a nonempty list in {spec}"
            )
        for row in rows:
            fields = {"entry", "file", "symbol", "upstream_revision", "use"}
            if (
                not isinstance(row, dict)
                or row.keys() != fields
                or not all(
                    isinstance(value, str) and value.strip() for value in row.values()
                )
            ):
                raise ValueError(f"Invalid upstream component declaration in {spec}")
            entry, file, symbol = (row[key] for key in ("entry", "file", "symbol"))
            for source in (entry, file):
                local = (root / source).resolve()
                if path.name == "SPEC.md":
                    in_model = (
                        path.parent != root / "vllm_metax/models"
                        and local.is_relative_to(path.parent)
                    )
                else:
                    name = path.name.removesuffix(".SPEC.md") + ".py"
                    in_model = path.name.endswith(
                        ".SPEC.md"
                    ) and local == path.with_name(name)
                if (
                    not in_model
                    or local.suffix != ".py"
                    or local.relative_to(root).as_posix() != source
                ):
                    raise ValueError(
                        f"Declared source is outside its model spec: {source}"
                    )
            module, _, _ = symbol.rpartition(".")
            family = (
                Path(entry)
                .relative_to("vllm_metax/models")
                .parts[0]
                .removesuffix(".py")
            )
            same_family = (
                f"vllm.models.{family}",
                f"vllm.model_executor.models.{family}",
            )
            if (
                entry not in entries
                or not concrete(module)
                or not all(part.isidentifier() for part in symbol.split("."))
                or any(
                    module == name or module.startswith(name + ".")
                    for name in same_family
                )
            ):
                raise ValueError(
                    f"Expected a cross-model symbol at a reviewed entry in {spec}"
                )
            per_entry = declarations.setdefault(entry, {})
            key = (file, symbol)
            if key in per_entry:
                raise ValueError(f"Duplicate component declaration for {entry}: {key}")
            per_entry[key] = {**row, "spec": spec}
    if declarations:
        if upstream_source is None:
            raise ValueError(
                "Upstream component reuse needs the selected vLLM checkout"
            )
        revision = subprocess.check_output(
            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
            cwd=upstream_source,
            text=True,
        ).strip()
        if any(
            row["upstream_revision"] != revision
            for declared in declarations.values()
            for row in declared.values()
        ):
            raise ValueError(
                "Upstream component revision differs from the selected vLLM checkout"
            )
    return declarations


def _framework_reviews(root, entries, review_files, upstream_source):
    reviews = {}
    revision = None
    for path in review_files:
        path = Path(path).resolve()
        if path.is_relative_to(root):
            raise ValueError(
                "Framework import review files must stay outside the source tree"
            )
        row = json.loads(path.read_text())
        if (
            not isinstance(row, dict)
            or set(row) != {"entry", "upstream_revision", "closure_sha256", "sites"}
            or not isinstance(row["entry"], str)
            or not isinstance(row["closure_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", row["closure_sha256"])
            or not isinstance(row["sites"], list)
            or not row["sites"]
        ):
            raise ValueError(f"Invalid framework dynamic import review in {path}")
        if upstream_source is None:
            raise ValueError(
                "Framework dynamic review needs the selected vLLM checkout"
            )
        if revision is None:
            revision = subprocess.check_output(
                ["git", "rev-parse", "--verify", "HEAD^{commit}"],
                cwd=upstream_source,
                text=True,
            ).strip()
        if row["upstream_revision"] != revision:
            raise ValueError(
                f"Framework dynamic review targets a different vLLM revision in {path}"
            )
        entry = row["entry"]
        entry_path = (root / entry).resolve()
        if (
            entry not in entries
            or not entry_path.is_relative_to(root / "vllm_metax/models")
            or entry_path.suffix != ".py"
            or entry_path.relative_to(root).as_posix() != entry
            or entry in reviews
        ):
            raise ValueError(
                f"Framework dynamic review needs one audited model entry in {path}"
            )
        sites = {}
        for site in row["sites"]:
            if (
                not isinstance(site, dict)
                or set(site) != {"file", "line", "reason"}
                or not isinstance(site["file"], str)
                or type(site["line"]) is not int
                or site["line"] < 1
                or not isinstance(site["reason"], str)
                or not site["reason"].strip()
            ):
                raise ValueError(f"Invalid reviewed framework import site in {path}")
            file = (root / site["file"]).resolve()
            key = (site["file"], site["line"])
            if (
                not file.is_relative_to(root / "vllm_metax/patch")
                or file.suffix != ".py"
                or file.relative_to(root).as_posix() != site["file"]
                or key in sites
            ):
                raise ValueError(
                    f"Reviewed dynamic imports must name unique framework patch sites in {path}"
                )
            sites[key] = site["reason"]
        reviews[entry] = (row["closure_sha256"], sites, path.name)
    return reviews


def _apply_framework_review(result, review):
    digest, sites, path = review
    actual = hashlib.sha256(
        json.dumps(result["files"], sort_keys=True).encode()
    ).hexdigest()
    if digest != actual:
        raise ValueError(f"Framework dynamic import review is stale in {path}")
    pending = []
    reviewed = []
    for item in result["dynamic_imports"]:
        key = (item["file"], item["line"])
        if key in sites:
            reviewed.append({**item, "reason": sites[key], "review_file": path})
        else:
            pending.append(item)
    if set(sites) != {(item["file"], item["line"]) for item in reviewed}:
        raise ValueError(
            f"Framework dynamic import review contains unreachable sites in {path}"
        )
    result["dynamic_imports"] = pending
    result["reviewed_framework_dynamic_imports"] = reviewed


def audit(root, entries, specs=(), upstream_source=None, framework_reviews=()):
    root = Path(root).resolve()
    declarations = _declarations(root, entries, specs, upstream_source)
    reviews = _framework_reviews(root, entries, framework_reviews, upstream_source)
    if not declarations and not reviews:
        return _audit(root, entries, {})
    report = _audit(root, (), {})
    boundaries = set()
    report["reviewed_framework_dynamic_imports"] = []
    for entry in dict.fromkeys(entries):
        result = _audit(root, (entry,), declarations.get(entry, {}))
        if entry in reviews:
            _apply_framework_review(result, reviews[entry])
        report["files"].update(result["files"])
        boundaries.update(result["public_framework_boundaries"])
        for key in (
            "prohibited_imports",
            "dynamic_imports",
            "upstream_component_reuse",
            "reviewed_framework_dynamic_imports",
        ):
            report[key].extend({"entry": entry, **item} for item in result.get(key, []))
    report["public_framework_boundaries"] = sorted(boundaries)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--entry", action="append", required=True)
    parser.add_argument("--spec", action="append", default=[])
    parser.add_argument("--upstream-source", type=Path)
    parser.add_argument("--framework-review", type=Path, action="append", default=[])
    args = parser.parse_args()
    try:
        report = audit(
            args.source,
            args.entry,
            args.spec,
            args.upstream_source,
            args.framework_review,
        )
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))
    raise SystemExit(
        1 if report["prohibited_imports"] or report["dynamic_imports"] else 0
    )
