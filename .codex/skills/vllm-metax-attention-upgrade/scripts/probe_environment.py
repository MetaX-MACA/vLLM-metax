#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Read-only attention source and component inventory; use the target Python."""

import argparse
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import unquote, urlsplit


def git(source, *args):
    result = subprocess.run(
        ["git", "-C", str(source), *args], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def manifest(root):
    if root is None or not root.is_dir():
        return None
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }


def compare(left, right):
    if left is None or right is None:
        return {"status": "unresolved"}
    generated = {"_version.py", "version.py"} & (left.keys() | right.keys())
    changed = sorted(k for k in left.keys() & right.keys() if left[k] != right[k])
    missing = sorted(left.keys() - right.keys())
    extra = sorted(right.keys() - left.keys())
    return {
        "status": "identical" if left == right else "different",
        "matching_files": sum(left[k] == right[k] for k in left.keys() & right.keys()),
        "changed_files": changed,
        "local_only_files": missing,
        "installed_only_files": extra,
        "version_file_differences": sorted(generated & set(changed + missing + extra)),
        "note": "Version files are reported, not automatically excluded as harmless.",
    }


def provenance(dist):
    raw = dist.read_text("direct_url.json")
    if not raw:
        return None
    data = json.loads(raw)
    url = urlsplit(data.get("url", ""))
    return {
        "scheme": url.scheme,
        "local_source_or_archive": unquote(url.path) if url.scheme == "file" else None,
        "editable": data.get("dir_info", {}).get("editable", False),
        "vcs_commit": data.get("vcs_info", {}).get("commit_id"),
        "note": "Remote URLs are omitted; a local build path is not a runtime binding.",
    }


def package_report(module, source):
    source = source.resolve()
    local_root = source / module
    local_manifest = manifest(local_root)
    local = {
        "checkout": str(source),
        "package_root": str(local_root),
        "exists": local_root.is_dir(),
        "git_root": git(source, "rev-parse", "--show-toplevel"),
        "commit": git(source, "rev-parse", "HEAD"),
        "status": git(source, "status", "--short", "--untracked-files=normal"),
        "python_file_count": len(local_manifest)
        if local_manifest is not None
        else None,
        "manifest_sha256": hashlib.sha256(
            json.dumps(local_manifest, sort_keys=True).encode()
        ).hexdigest()
        if local_manifest is not None
        else None,
    }
    try:
        spec = importlib.util.find_spec(module)
        effective = {
            "origin": spec.origin if spec else None,
            "search_locations": [
                str(Path(p).resolve()) for p in (spec.submodule_search_locations or [])
            ]
            if spec
            else [],
        }
    except (ImportError, ValueError, AttributeError) as exc:
        effective = {"error": str(exc)}
    effective["includes_local_package"] = str(local_root.resolve()) in effective.get(
        "search_locations", []
    )
    distributions = []
    wanted = module.replace("_", "-").lower()
    for dist in metadata.distributions():
        name = (dist.metadata.get("Name") or "").replace("_", "-").lower()
        if name != wanted:
            continue
        files = list(dist.files or [])
        roots = {
            Path(dist.locate_file(f)).resolve().parent
            for f in files
            if str(f) == f"{module}/__init__.py"
        }
        metadata_paths = [
            str(Path(dist.locate_file(f)).resolve().parent)
            for f in files
            if str(f).endswith((".dist-info/METADATA", ".egg-info/PKG-INFO"))
        ]
        distributions.append(
            {
                "name": dist.metadata["Name"],
                "version": dist.version,
                "installation_root": str(Path(dist.locate_file("")).resolve()),
                "metadata_paths": metadata_paths,
                "metadata_note": (
                    "Source-tree egg-info may be visible without a wheel installation."
                    if any(p.endswith(".egg-info") for p in metadata_paths)
                    else "Check provenance for editable installs and build origin."
                ),
                "installer": (dist.read_text("INSTALLER") or "").strip(),
                "provenance": provenance(dist),
                "package_trees": [
                    {
                        "root": str(root),
                        "selected_by_import_probe": str(root)
                        in effective.get("search_locations", []),
                        "comparison_to_local": compare(local_manifest, manifest(root)),
                    }
                    for root in sorted(roots)
                ],
                "extension_paths": [
                    str(Path(dist.locate_file(f)).resolve())
                    for f in files
                    if str(f).endswith((".so", ".pyd", ".dylib"))
                    and str(f).startswith(module + "/")
                ],
                "mapping_note": None
                if roots
                else "No package tree in distribution records; inspect editable/path mappings.",
            }
        )
    return {
        "local": local,
        "effective_import_probe": effective,
        "installed_distributions": distributions,
    }


def component_report():
    """Inventory candidates without importing GPU libraries or claiming API support."""
    modules = {}
    for name in (
        "deep_gemm",
        "flash_attn",
        "flash_attn_3",
        "flash_mla",
        "mcoplib",
        "torch",
        "triton",
    ):
        try:
            spec = importlib.util.find_spec(name)
            modules[name] = {
                "origin": spec.origin if spec else None,
                "search_locations": list(spec.submodule_search_locations or [])
                if spec
                else [],
            }
        except (ImportError, ValueError, AttributeError) as exc:
            modules[name] = {"error": str(exc)}
    distributions = []
    for dist in metadata.distributions():
        name = dist.metadata.get("Name", "")
        normalized = name.lower().replace("-", "_")
        if not any(
            normalized.startswith(prefix)
            for prefix in (
                "deep_gemm",
                "flash_attn",
                "flash_mla",
                "flashmla",
                "mcoplib",
                "torch",
                "triton",
            )
        ):
            continue
        files = list(dist.files or [])
        distributions.append(
            {
                "name": name,
                "version": dist.version,
                "installation_root": str(Path(dist.locate_file("")).resolve()),
                "metadata_paths": [
                    str(Path(dist.locate_file(f)).resolve().parent)
                    for f in files
                    if str(f).endswith((".dist-info/METADATA", ".egg-info/PKG-INFO"))
                ],
                "provenance": provenance(dist),
                "extension_paths": [
                    str(Path(dist.locate_file(f)).resolve())
                    for f in files
                    if str(f).endswith((".so", ".pyd", ".dylib"))
                ],
            }
        )
    return {
        "top_level_resolution": modules,
        "distribution_candidates": distributions,
        "note": (
            "Distribution names need not equal import names. Verify actual callable and "
            "loaded extension origins in the test process; these are candidates, not "
            "proof of dispatch, API compatibility, or numerical correctness."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--metax-source", type=Path, required=True)
    parser.add_argument("--runtime-cwd", type=Path, default=Path.cwd())
    args = parser.parse_args()
    # Resolve relative checkout paths before modeling the requested working directory.
    vllm_source, metax_source = args.vllm_source.resolve(), args.metax_source.resolve()
    runtime_cwd = args.runtime_cwd.resolve()
    os.chdir(runtime_cwd)
    sys.path[0] = str(runtime_cwd)
    cfg = Path(sys.prefix) / "pyvenv.cfg"
    dependencies = {}
    for name in ("torch", "triton", "transformers"):
        try:
            dependencies[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            dependencies[name] = None
    report = {
        "python": {
            "executable": sys.executable,
            "executable_realpath": str(Path(sys.executable).resolve()),
            "version": sys.version,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "is_venv": sys.prefix != sys.base_prefix,
            "pyvenv_cfg": cfg.read_text() if cfg.exists() else None,
            "runtime_cwd": str(runtime_cwd),
            "path_hints": {
                k: os.environ.get(k)
                for k in ("VIRTUAL_ENV", "CONDA_PREFIX", "PYTHONPATH")
            },
            "modeled_sys_path": sys.path,
        },
        "dependencies": dependencies,
        "attention_components": component_report(),
        "vllm": package_report("vllm", vllm_source),
        "vllm_metax": package_report("vllm_metax", metax_source),
        "limitations": [
            "Top-level resolution models python -c/-m; verify origins in the actual test process.",
            "No package imports or binary ABI validation were performed.",
            "Successful collection is not a compatibility verdict; review every mismatch.",
        ],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
