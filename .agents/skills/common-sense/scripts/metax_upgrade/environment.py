# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Independent, stdlib-only inventory and fresh collect_env diagnostics.

Upgrade facts are collected without importing vLLM, MetaX or GPU packages. Each
full inventory executes the ordinary collect_env script as supplementary text.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import regex as re
import shutil
import signal
import subprocess
import sys
from urllib.parse import unquote, urlsplit


def _runtime_environment() -> dict:
    # NOTE(metax): Keep an explicit dispatch/ABI allowlist, never arbitrary env dumps.
    keys = (
        "VLLM_PLUGINS",
        "VLLM_TARGET_DEVICE",
        "VLLM_ATTENTION_BACKEND",
        "VLLM_USE_V1",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "CUDA_HOME",
        "VLLM_MCCL_SO_PATH",
        "USE_PRECOMPILED_KERNEL",
        "MACA_VLLM_ENABLE_MCTLASS_FUSED_MOE",
        "VLLM_METAX_OPTIMIZED_DP_ALL2ALL",
        "VLLM_METAX_ENABLE_FA_SPLIT_FORWARD",
        "VLLM_METAX_SUPPORTS_FP8",
        "VLLM_METAX_USE_FP8_WO_A",
        "VLLM_METAX_USE_QUANT_MAIN_PROJ",
        "VLLM_METAX_USE_SGL_FUSED_MOE_GROUPED_TOPK",
        "VLLM_METAX_USE_FLASHINFER_SAMPLER",
        "VLLM_USE_FLASHINFER_SAMPLER",
        "USE_VLLM_TRITON_EXPERT",
    )
    return {key: os.environ.get(key) for key in keys}


def _conda_selection(
    prefix: str, base_prefix: str, inherited_prefix: str | None
) -> dict:
    matches = (
        bool(inherited_prefix)
        and Path(inherited_prefix).resolve() == Path(prefix).resolve()
    )
    return {
        "selected_prefix": prefix,
        "is_selected_conda": Path(prefix, "conda-meta").is_dir()
        or (prefix == base_prefix and matches),
        "inherited_prefix": inherited_prefix,
        "inherited_prefix_matches_selected": matches,
        "note": "Selected Python prefix is authoritative; an inherited parent Conda prefix is provenance only.",
    }


def normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _sanitize(value: str) -> str:
    # NOTE(metax): Collector output must not persist arbitrary environment values.
    value = re.split(r"(?im)^\s*Environment Variables\s*$", value, maxsplit=1)[0]
    for key, secret in os.environ.items():
        if (
            re.search(
                r"TOKEN|PASS|SECRET|API_KEY|ACCESS_KEY|CREDENTIAL|AUTH", key, re.I
            )
            and len(secret) >= 4
        ):
            value = value.replace(secret, "[REDACTED]")
    value = re.sub(r"(https?://)[^\s/@]+(?::[^\s/@]*)?@", r"\1[REDACTED]@", value)
    value = re.sub(
        r"(?i)((?:token|password|secret|api[_-]?key|access[_-]?key)[\w-]*\s*[=:]\s*)[^\s,;]+",
        r"\1[REDACTED]",
        value,
    )
    return value[:250_000]


def _run(command: list[str], cwd: Path, timeout: float, env=None) -> dict:
    result = {"command": command}
    try:
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=os.name == "posix",
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            status = "ok" if proc.returncode == 0 else "error"
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(proc.pid, signal.SIGKILL)
            else:
                proc.kill()
            stdout, stderr = proc.communicate()
            status = "timeout"
        result.update(
            status=status,
            returncode=proc.returncode,
            stdout=_sanitize(stdout),
            stderr=_sanitize(stderr),
        )
    except (OSError, ValueError) as exc:
        result.update(
            status="missing" if isinstance(exc, FileNotFoundError) else "error",
            returncode=None,
            stdout="",
            stderr=_sanitize(str(exc)),
        )
    return result


def _inventory() -> tuple[list[dict], list[str]]:
    packages, issues = [], []
    for dist in metadata.distributions():
        try:
            name = dist.metadata.get("Name") or ""
            if not name:
                issues.append(
                    "A distribution has no Name metadata; inspect package inventory."
                )
            files = list(dist.files or [])
            record = {
                "name": name,
                "normalized_name": normalize_name(name),
                "version": dist.version,
                "installation_root": str(Path(dist.locate_file("")).resolve()),
                "metadata_locations": sorted(
                    {
                        str(Path(dist.locate_file(f)).resolve().parent)
                        for f in files
                        if str(f).endswith(
                            (".dist-info/METADATA", ".egg-info/PKG-INFO")
                        )
                    }
                ),
                "installer": (dist.read_text("INSTALLER") or "").strip(),
                "provenance": None,
            }
            raw = dist.read_text("direct_url.json")
            if raw:
                data = json.loads(raw)
                url = urlsplit(data.get("url", ""))
                record["provenance"] = {
                    "scheme": url.scheme,
                    "local_path": unquote(url.path) if url.scheme == "file" else None,
                    "editable": data.get("dir_info", {}).get("editable", False),
                    "vcs_commit": data.get("vcs_info", {}).get("commit_id"),
                    "archive_hashes": data.get("archive_info", {}).get("hashes", {}),
                }
            packages.append(record)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            issues.append(
                "Distribution metadata could not be read: " + _sanitize(str(exc))
            )
    packages.sort(
        key=lambda d: (
            d["normalized_name"],
            d["version"],
            d["installation_root"],
            d["metadata_locations"],
        )
    )
    return packages, issues


def _binary_identity(packages: list[dict], selected_names: set[str]) -> dict:
    """Track recorded native files without importing or hashing large binaries."""
    installed = {package["normalized_name"] for package in packages}
    selected = {normalize_name(name) for name in selected_names} & installed
    distributions, issues = [], []
    for dist in metadata.distributions():
        try:
            name = normalize_name(dist.metadata.get("Name") or "")
            if name not in selected:
                continue
            raw_record = dist.read_text("RECORD")
            record = {
                "normalized_name": name,
                "version": dist.version,
                "installation_root": str(Path(dist.locate_file("")).resolve()),
                "record_sha256": hashlib.sha256(raw_record.encode()).hexdigest()
                if raw_record is not None
                else None,
                "native_files": [],
            }
            files = dist.files
            record["distribution_file_records_available"] = files is not None
            for entry in files or []:
                if not re.search(
                    r"\.so(?:\.[^/]+)?$|\.pyd$|\.dylib$", str(entry), re.I
                ):
                    continue
                path = Path(dist.locate_file(entry)).resolve()
                native = {"path": str(path)}
                try:
                    stat = path.stat()
                    native.update(
                        size=stat.st_size,
                        mtime_ns=stat.st_mtime_ns,
                        inode=stat.st_ino,
                        device=stat.st_dev,
                    )
                except OSError as exc:
                    native["stat_error"] = exc.errno
                    issues.append(
                        f"Cannot stat native component {path}: errno {exc.errno}"
                    )
                record["native_files"].append(native)
            record["native_files"].sort(key=lambda native: native["path"])
            distributions.append(record)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            issues.append("Native component inventory failed: " + _sanitize(str(exc)))
    distributions.sort(
        key=lambda record: (
            record["normalized_name"],
            record["version"],
            record["installation_root"],
        )
    )
    return {
        "distributions": distributions,
        "issues": sorted(set(issues)),
        "limitations": (
            "Stat tokens and distribution RECORD hashes detect ordinary replacements, "
            "not full binary integrity or ABI compatibility. Only distribution-recorded "
            "native files are inventoried; actual tested extension paths need separate evidence."
        ),
    }


def read_requirements(path: Path, packages: list[dict]) -> dict:
    """Handle local includes and exact pins; never silently evaluate complex specs."""
    files, entries, issues, active = {}, [], [], set()
    installed = {}
    for package in packages:
        installed.setdefault(package["normalized_name"], []).append(package["version"])

    def visit(current: Path, constraint: bool = False):
        current = current.resolve()
        if current in active:
            issues.append(f"Requirements include cycle at {current}")
            return
        try:
            content = current.read_bytes()
        except OSError as exc:
            files[str(current)] = None
            issues.append(f"Cannot read requirements {current}: {_sanitize(str(exc))}")
            return
        files[str(current)] = hashlib.sha256(content).hexdigest()
        active.add(current)
        lines = (
            content.decode("utf-8", errors="replace").replace("\\\n", " ").splitlines()
        )
        for lineno, raw in enumerate(lines, 1):
            line = re.sub(r"\s+#.*$", "", raw).strip()
            if not line or line.startswith("#"):
                continue
            entry = {
                "file": str(current),
                "line": lineno,
                "requirement": _sanitize(line),
                "constraint": constraint,
            }
            include = re.match(r"^(?:-r\s*|--requirement(?:=|\s+))(.+)$", line)
            constraints = re.match(r"^(?:-c\s*|--constraint(?:=|\s+))(.+)$", line)
            if include or constraints:
                target = (include or constraints).group(1).strip().strip("\"'")
                if "://" in target or "${" in target:
                    entry["status"] = "unsupported_include"
                    issues.append(
                        f"Requirement include needs manual review: {current}:{lineno}"
                    )
                else:
                    entry["status"] = "include"
                    visit(current.parent / target, bool(constraints) or constraint)
                entries.append(entry)
                continue
            if line.startswith("-"):
                option = line.split("=", 1)[0].split()[0]
                entry["status"] = (
                    "option"
                    if option
                    in {
                        "--index-url",
                        "--extra-index-url",
                        "--trusted-host",
                        "--find-links",
                        "--no-index",
                        "--pre",
                        "--prefer-binary",
                        "--only-binary",
                        "--no-binary",
                    }
                    else "unsupported_option"
                )
                if entry["status"] == "unsupported_option":
                    issues.append(
                        f"Unsupported requirements option: {current}:{lineno}"
                    )
                entries.append(entry)
                continue
            pin = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9_.-]*)\s*==\s*([^\s;*]+)", line)
            if pin:
                name, expected = pin.groups()
                versions = installed.get(normalize_name(name), [])
                status = (
                    "missing"
                    if not versions
                    else "ambiguous"
                    if len(versions) > 1
                    else ("match" if versions[0] == expected else "mismatch")
                )
                # Constraints restrict a package if installed; they do not require installation.
                if constraint and not versions:
                    status = "constraint_not_installed"
                entry.update(
                    normalized_name=normalize_name(name),
                    expected_version=expected,
                    installed_versions=versions,
                    status=status,
                )
                if status not in {"match", "constraint_not_installed"}:
                    issues.append(
                        f"Requirement {name}=={expected}: {status}; installed {versions}"
                    )
            else:
                entry["status"] = "unsupported_specifier"
                issues.append(
                    f"Requirement needs manual evaluation (markers/ranges/extras/URLs): {current}:{lineno}"
                )
            entries.append(entry)
        active.remove(current)

    visit(path)
    return {
        "root": str(path.resolve()),
        "files": files,
        "entries": entries,
        "issues": issues,
        "comparison": "Literal exact-pin comparison; unsupported forms are never assumed satisfied.",
    }


def _source_identity(report: dict) -> dict:
    result = {}
    for name in ("vllm", "vllm_metax"):
        info = report.get(name, {})
        local = info.get("local", {})
        result[name] = {
            "checkout": local.get("checkout"),
            "package_root": local.get("package_root"),
            "exists": local.get("exists"),
            "effective_import_probe": info.get("effective_import_probe"),
        }
        if name == "vllm":
            result[name].update(
                commit=local.get("commit"), manifest_sha256=local.get("manifest_sha256")
            )
    return result


def _gpu_identity(output: str) -> list[dict]:
    models = []
    for line in output.splitlines():
        match = re.match(r"\s*GPU(?:#|\s+)(\d+)\s*:?\s+(.+)", line)
        if match:
            description = re.sub(r"\b(?:Available|Unavailable)\b", "", match.group(2))
            models.append(
                {
                    "index": int(match.group(1)),
                    "description": " ".join(description.split()),
                }
            )
    return models


def _sdk(cwd: Path, timeout: float) -> tuple[dict, dict]:
    roots = [
        Path(value)
        for key in ("MACA_PATH", "MACA_HOME")
        if (value := os.environ.get(key))
    ]
    roots.append(Path("/opt/maca"))
    candidates = set()
    for root in roots:
        candidates.update(
            root / name for name in ("Version.txt", "version.txt", "version")
        )
        candidates.update(root.glob("version*"))
    versions, errors = {}, []
    for path in sorted(candidates):
        if path.is_file():
            try:
                versions[str(path.resolve())] = _sanitize(
                    path.read_text()[:65536]
                ).strip()
            except OSError as exc:
                errors.append(_sanitize(str(exc)))
    mxcc = shutil.which("mxcc")
    if mxcc is None:
        for root in roots:
            for relative in ("bin/mxcc", "mxgpu_llvm/bin/mxcc"):
                if (root / relative).is_file():
                    mxcc = str(root / relative)
                    break
            if mxcc is not None:
                break
    compiler = _run([mxcc or "mxcc", "--version"], cwd, timeout)
    compiler_version = []
    sdk_versions = sorted(
        {
            match
            for value in versions.values()
            for match in re.findall(
                r"(?i)(?:version\s*[:=]?\s*)?([0-9]+(?:\.[0-9A-Za-z_-]+)+)", value
            )
        }
    )
    if compiler["status"] == "ok":
        compiler_version = re.findall(
            r"(?i)(?:version|release)\s*[:=]?\s*([0-9]+(?:\.[0-9A-Za-z_-]+)+)",
            compiler["stdout"] + "\n" + compiler["stderr"],
        )
        sdk_versions.extend(
            re.findall(
                r"(?i)MACA(?:\s+SDK)?\s+version\s*[:=]?\s*([0-9]+(?:\.[0-9A-Za-z_-]+)+)",
                compiler["stdout"] + "\n" + compiler["stderr"],
            )
        )
    identity = {
        "version_files": versions,
        "compiler_path": mxcc,
        "compiler_versions": compiler_version,
        "sdk_versions": sorted(set(sdk_versions)),
        "declared_version": os.environ.get("MACA_AI_VERSION"),
        "paths": {key: os.environ.get(key) for key in ("MACA_PATH", "MACA_HOME")},
    }
    return identity, {"compiler": compiler, "version_read_errors": errors}


def fingerprint(report: dict) -> str:
    """Stable environment identity, excluding timestamps and diagnostic output."""
    payload = json.dumps(report["identity"], sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _git(source, *args):
    result = _run(["git", "-C", str(source), *args], Path.cwd(), 15)
    return result["stdout"].strip() if result["status"] == "ok" else None


def _manifest(root):
    if root is None or not root.is_dir():
        return None
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }


def _compare(left, right):
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


def _provenance(dist):
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


def _package_report(module, source):
    source = source.resolve() if source is not None else None
    local_root = source / module if source is not None else None
    local_manifest = _manifest(local_root)
    local = {
        "checkout": str(source) if source is not None else None,
        "package_root": str(local_root) if local_root is not None else None,
        "exists": local_root.is_dir() if local_root is not None else None,
        "git_root": _git(source, "rev-parse", "--show-toplevel")
        if source is not None
        else None,
        "commit": _git(source, "rev-parse", "HEAD") if source is not None else None,
        "status": _git(source, "status", "--short", "--untracked-files=normal")
        if source is not None
        else None,
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
    effective["includes_local_package"] = (
        str(local_root.resolve()) in effective.get("search_locations", [])
        if local_root is not None
        else None
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
                "provenance": _provenance(dist),
                "package_trees": [
                    {
                        "root": str(root),
                        "selected_by_import_probe": str(root)
                        in effective.get("search_locations", []),
                        "comparison_to_local": _compare(
                            local_manifest, _manifest(root)
                        ),
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


def _component_report():
    """Inventory candidates without importing GPU libraries or claiming API support."""
    modules = {}
    for name in (
        "compressed_tensors",
        "flashinfer",
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
                "compressed_tensors",
                "flashinfer",
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
                "provenance": _provenance(dist),
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


def _discover_checkout(module: str) -> Path | None:
    """Find checkouts from local files or distribution metadata without importing."""
    candidates = []
    if module == "vllm_metax":
        candidates.append(Path(__file__).resolve().parents[5])
    else:
        candidates.append(Path(__file__).resolve().parents[5].parent / "vllm")
    wanted = normalize_name(module)
    for dist in metadata.distributions():
        if normalize_name(dist.metadata.get("Name") or "") != wanted:
            continue
        try:
            raw = dist.read_text("direct_url.json")
            data = json.loads(raw) if raw else {}
            url = urlsplit(data.get("url", ""))
            if url.scheme == "file":
                candidates.append(Path(unquote(url.path)))
        except (OSError, ValueError, TypeError, KeyError):
            continue
    for candidate in candidates:
        if (candidate / module).is_dir() and (
            (candidate / "pyproject.toml").is_file() or (candidate / ".git").exists()
        ):
            return candidate.resolve()
    return None


def _source_probe(
    vllm_source: Path | None, metax_source: Path | None, runtime_cwd: Path
) -> dict:
    """Called only in a child so cwd/sys.path modeling cannot leak into callers."""
    os.chdir(runtime_cwd)
    sys.path.insert(0, str(runtime_cwd))
    cfg = Path(sys.prefix) / "pyvenv.cfg"
    dependencies = {}
    for name in ("torch", "triton", "transformers"):
        try:
            dependencies[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            dependencies[name] = None
    return {
        "python": {
            "executable": sys.executable,
            "executable_realpath": str(Path(sys.executable).resolve()),
            "version": sys.version,
            "prefix": sys.prefix,
            "base_prefix": sys.base_prefix,
            "is_venv": sys.prefix != sys.base_prefix,
            "pyvenv_cfg": _sanitize(cfg.read_text()) if cfg.is_file() else None,
            "runtime_cwd": str(runtime_cwd),
            "path_hints": {
                key: os.environ.get(key)
                for key in ("VIRTUAL_ENV", "CONDA_PREFIX", "PYTHONPATH")
            },
            "modeled_sys_path": sys.path,
        },
        "dependencies": dependencies,
        "components": _component_report(),
        "vllm": _package_report("vllm", vllm_source),
        "vllm_metax": _package_report("vllm_metax", metax_source),
        "limitations": [
            "Top-level resolution models python -c/-m; verify origins in the actual test process.",
            "No package imports or binary ABI validation were performed.",
            "Successful collection is not a compatibility verdict; review every mismatch.",
        ],
    }


def _system(cwd: Path, timeout: float, full: bool) -> tuple[dict, dict]:
    identity = {
        "platform": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "libc": list(platform.libc_ver()),
    }
    try:
        identity["os_release"] = platform.freedesktop_os_release()
    except (AttributeError, OSError):
        identity["os_release"] = {}
    details = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
    }
    try:
        details["cpu_affinity"] = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        details["cpu_affinity"] = None
    commands = {
        "cpu": ["lscpu"],
        "gcc": ["gcc", "--version"],
        "clang": ["clang", "--version"],
        "cmake": ["cmake", "--version"],
    }
    details["commands"] = {
        name: _run(command, cwd, timeout)
        if full
        else {"status": "skipped", "note": "Identity-only refresh."}
        for name, command in commands.items()
    }
    return identity, details


def collect_environment(
    vllm_source: Path | None = None,
    metax_source: Path | None = None,
    runtime_cwd: Path | None = None,
    requirements: Path | None = None,
    timeout: float = 30,
    include_collect_env: bool = True,
) -> dict:
    """Collect the selected Python environment without importing GPU packages.

    Explicit source paths are upgrade obligations. Omitted paths are discovered
    when possible, and an installed wheel does not require a source checkout.
    collect_env text is supplementary and freshly collected for every full snapshot.
    Identity-only refreshes independently recollect facts without that subprocess.
    """
    runtime_cwd = Path(runtime_cwd or Path.cwd()).resolve()
    explicit_sources = {
        "vllm": vllm_source is not None,
        "vllm_metax": metax_source is not None,
    }
    vllm_source = (
        Path(vllm_source).resolve()
        if vllm_source is not None
        else _discover_checkout("vllm")
    )
    metax_source = (
        Path(metax_source).resolve()
        if metax_source is not None
        else _discover_checkout("vllm_metax")
    )
    issues = []
    packages, inventory_issues = _inventory()
    issues.extend(inventory_issues)
    requirements_root = (
        Path(requirements).resolve()
        if requirements is not None
        else (metax_source or Path(__file__).resolve().parents[5])
        / "requirements/maca_private.txt"
    )
    required = read_requirements(requirements_root, packages)
    required["status"] = (
        "ok"
        if all(value is not None for value in required["files"].values())
        else "missing"
    )
    if required["status"] == "missing":
        required["note"] = (
            "Provide --requirements /path/to/requirements/maca_private.txt when the "
            "requirements file is not included in an installed wheel."
        )
    issues.extend(required["issues"])
    native_files = _binary_identity(
        packages,
        {
            entry["normalized_name"]
            for entry in required["entries"]
            if "normalized_name" in entry
        }
        | {"vllm", "vllm-metax"},
    )
    issues.extend(native_files["issues"])
    cfg = Path(sys.prefix) / "pyvenv.cfg"
    cfg_text = _sanitize(cfg.read_text()) if cfg.is_file() else None
    conda_selection = _conda_selection(
        sys.prefix, sys.base_prefix, os.environ.get("CONDA_PREFIX")
    )
    python = {
        "executable": sys.executable,
        "executable_realpath": str(Path(sys.executable).resolve()),
        "version": sys.version,
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "is_venv": sys.prefix != sys.base_prefix,
        "environment_manager": (
            "uv"
            if cfg_text and re.search(r"(?m)^uv\s*=", cfg_text)
            else "conda"
            if conda_selection["is_selected_conda"]
            else "venv"
            if sys.prefix != sys.base_prefix
            else "system"
        ),
        "runtime_cwd": str(runtime_cwd),
        "pyvenv_cfg": cfg_text,
        "path_hints": {
            key: os.environ.get(key)
            for key in (
                "VIRTUAL_ENV",
                "CONDA_PREFIX",
                "CONDA_DEFAULT_ENV",
                "PYTHONPATH",
            )
        },
    }
    # NOTE(metax): Use this stdlib file directly; never initialize the plugin.
    probe_args = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--source-probe",
        "--runtime-cwd",
        str(runtime_cwd),
    ]
    for flag, source in (
        ("--vllm-source", vllm_source),
        ("--metax-source", metax_source),
    ):
        if source is not None:
            probe_args.extend([flag, str(source)])
    probe_command = _run(probe_args, runtime_cwd, timeout)
    probe = {}
    if probe_command["status"] == "ok":
        try:
            probe = json.loads(probe_command["stdout"])
        except ValueError:
            issues.append("Source/runtime probe returned invalid or truncated JSON.")
    else:
        issues.append(
            "Source/runtime correspondence probe failed: " + probe_command["status"]
        )
    for name in ("vllm", "vllm_metax"):
        info = probe.get(name, {})
        if explicit_sources[name]:
            if not info.get("local", {}).get("exists"):
                issues.append(
                    f"{name} source package is missing or could not be inventoried."
                )
            if not info.get("effective_import_probe", {}).get("includes_local_package"):
                issues.append(
                    f"{name} runtime resolution does not select the supplied source package."
                )
    sdk, sdk_details = _sdk(runtime_cwd, timeout)
    if not sdk["sdk_versions"]:
        issues.append(
            "MACA SDK version was not independently collected from SDK files or mxcc."
        )
    if sdk_details["compiler"]["status"] != "ok":
        issues.append(
            "MACA compiler probe failed: " + sdk_details["compiler"]["status"]
        )
    smi = shutil.which("mx-smi") or "mx-smi"
    gpu_details = {
        name: _run([smi, *args], runtime_cwd, timeout)
        for name, args in (
            ("models", ["-L"]),
            ("version", ["--version"]),
            ("topology", ["topo", "-m"]),
            ("driver", []),
        )
    }
    models = (
        _gpu_identity(gpu_details["models"]["stdout"])
        if gpu_details["models"]["status"] == "ok"
        else []
    )
    if not models:
        issues.append(
            "MetaX GPU models/identities could not be collected with mx-smi -L."
        )
    for name, command in gpu_details.items():
        if command["status"] != "ok":
            issues.append(f"mx-smi {name} probe failed: {command['status']}")
    gpu = {
        "smi_path": shutil.which("mx-smi"),
        "models": models,
        "smi_version": gpu_details["version"]["stdout"].strip()
        if gpu_details["version"]["status"] == "ok"
        else None,
        "visibility": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "MACA_VISIBLE_DEVICES",
                "HIP_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
            )
        },
    }
    driver_output = gpu_details["driver"]["stdout"]
    for key, label in (
        ("driver_versions", "Driver Version"),
        ("runtime_versions", "MACA Version"),
        ("bios_versions", "BIOS Version"),
    ):
        gpu[key] = sorted(
            set(
                re.findall(
                    r"(?i)" + label + r"\s*:\s*([0-9]+(?:\.[0-9A-Za-z_-]+)+)",
                    driver_output,
                )
            )
        )
    conda_path = os.environ.get("CONDA_EXE") or shutil.which("conda")
    conda = (
        _run(
            [conda_path, "list", "--prefix", sys.prefix, "--json"], runtime_cwd, timeout
        )
        if conda_path
        else {"status": "missing", "note": "No conda executable found."}
    )
    conda_packages = []
    if conda["status"] == "ok":
        try:
            conda_packages = sorted(
                [
                    {
                        key: record.get(key)
                        for key in ("name", "version", "build_string", "build_number")
                    }
                    for record in json.loads(conda["stdout"])
                ],
                key=lambda record: record["name"],
            )
        except (ValueError, TypeError, KeyError):
            issues.append("conda list returned invalid package JSON.")
    elif conda_selection["is_selected_conda"]:
        issues.append(
            "Conda environment detected but selected-prefix conda inventory failed."
        )
    system_identity, system_details = _system(runtime_cwd, timeout, include_collect_env)
    report = {
        "schema_version": 1,
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "identity": {
            "python": python,
            "packages": packages,
            "conda_packages": conda_packages,
            "requirements": {"root": required["root"], "files": required["files"]},
            "sdk": sdk,
            "gpu": gpu,
            "source_mapping": _source_identity(probe),
            "runtime_environment": _runtime_environment(),
            "native_files": native_files,
            "system": system_identity,
        },
        "details": {
            "requirements": required,
            "source_probe": probe,
            "source_probe_command": probe_command,
            "sdk": sdk_details,
            "gpu": gpu_details,
            "conda": conda,
            "conda_selection": conda_selection,
            "system": system_details,
            "collection_profile": "full" if include_collect_env else "identity",
            "collect_env": {"status": "skipped", "note": "Identity-only refresh."},
        },
        "readiness": {
            "status": "review_required",
            "issues": sorted(set(issues)),
            "note": "Inventory is not compatibility evidence; full upgrades require an approved preparation plan and automatic installation verification.",
        },
    }

    if include_collect_env:
        report["details"]["collect_env"] = _collect_env(
            metax_source or Path(__file__).resolve().parents[5],
            runtime_cwd,
            timeout,
        )
    return report


def _collect_env(metax_source: Path, cwd: Path, timeout: float) -> dict:
    """Run fresh supplementary diagnostics; failures never suppress inventory."""
    metax_source, cwd = Path(metax_source).resolve(), Path(cwd).resolve()
    script = metax_source / "vllm_metax/collect_env.py"
    result = {
        "source": str(script),
        "source_hash": None,
        "collected_at": datetime.now(timezone.utc).isoformat(),
    }
    try:
        result["source_hash"] = hashlib.sha256(script.read_bytes()).hexdigest()
    except OSError as exc:
        result.update(
            status="missing" if isinstance(exc, FileNotFoundError) else "error",
            stdout="",
            stderr=_sanitize(str(exc)),
            note="Supplementary MetaX collector is unavailable; independent inventory is retained.",
        )
        return result
    env = os.environ.copy()
    env["VLLM_PLUGINS"] = ""
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result.update(_run([sys.executable, "-B", str(script)], cwd, timeout, env))
    result["note"] = (
        "Fresh supplementary collect_env output; plugins disabled, credentials "
        "and environment-variable section removed. Not compatibility evidence."
    )
    return result


def format_environment(report: dict) -> str:
    """Use the original collector text, with a small fallback if it failed."""
    diagnostic = report.get("details", {}).get("collect_env", {})
    if diagnostic.get("status") == "ok" and diagnostic.get("stdout"):
        return diagnostic["stdout"].rstrip("\n")
    identity = report.get("identity", {})
    python = identity.get("python", {})
    sdk = identity.get("sdk", {})
    models = identity.get("gpu", {}).get("models", [])
    lines = [
        "Environment inventory",
        f"Python: {python.get('executable', 'unknown')}",
        f"Environment: {python.get('environment_manager', 'unknown')} ({python.get('prefix', 'unknown')})",
        f"MACA SDK: {', '.join(sdk.get('sdk_versions', [])) or 'unknown'}",
        f"GPUs: {', '.join(model['description'] for model in models) or 'unknown'}",
        f"Installed packages: {len(identity.get('packages', []))}",
        f"collect_env: {diagnostic.get('status', 'unavailable')}",
    ]
    issues = report.get("readiness", {}).get("issues", [])
    if issues:
        lines.extend(["Review items:", *[f"  - {issue}" for issue in issues]])
    return "\n".join(lines)


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-probe", action="store_true", required=True)
    parser.add_argument("--vllm-source", type=Path)
    parser.add_argument("--metax-source", type=Path)
    parser.add_argument("--runtime-cwd", type=Path, default=Path.cwd())
    args = parser.parse_args()
    vllm_source = args.vllm_source.resolve() if args.vllm_source is not None else None
    metax_source = (
        args.metax_source.resolve() if args.metax_source is not None else None
    )
    print(
        json.dumps(
            _source_probe(vllm_source, metax_source, args.runtime_cwd.resolve()),
            indent=2,
        )
    )


if __name__ == "__main__":
    _main()
