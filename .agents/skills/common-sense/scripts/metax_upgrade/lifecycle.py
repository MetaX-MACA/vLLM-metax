# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Recoverable baseline capture and content identities without target imports."""

import hashlib
import importlib.machinery
import json
import os
from pathlib import Path, PurePosixPath
import regex as re
import shutil
import sys
import tempfile
import uuid
import zipfile
import zlib

if __package__:
    from . import state as store
else:
    import state as store


def _sync_directory(path):
    descriptor = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _checked_manifest(manifest):
    if not isinstance(manifest, dict):
        raise ValueError("Baseline source manifest must be an object")
    for name, content_hash in manifest.items():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or str(path) != name:
            raise ValueError("Unsafe baseline source path: " + name)
        if content_hash is not None and not re.fullmatch(r"[0-9a-f]{64}", content_hash):
            raise ValueError("Invalid baseline content hash: " + name)
    return manifest


def _archive_valid(path, manifest):
    expected = {name: value for name, value in manifest.items() if value is not None}
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)) or set(names) != set(expected):
                return False
            for name, expected_hash in expected.items():
                digest = hashlib.sha256()
                with archive.open(name) as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != expected_hash:
                    return False
        return True
    except (
        OSError,
        ValueError,
        RuntimeError,
        EOFError,
        zipfile.BadZipFile,
        zlib.error,
    ):
        return False


def prepare_baseline(run_dir, state):
    """Finish an initialization recorded *before* capture, without replacing history.

    The caller holds the run lock and has persisted ``source_manifest`` and
    ``initialization.phase = 'baseline'``. An existing valid final archive wins
    even if sources have since changed. Incomplete files remain available for
    inspection; rebuilding is permitted only from the frozen source identity.
    """
    run_dir = Path(run_dir)
    folder = run_dir / "baseline"
    folder.mkdir(parents=True, exist_ok=True)
    _sync_directory(run_dir)
    manifest_path = folder / "source-manifest.json"
    archive_path = folder / "metax-source.zip"
    manifest = _checked_manifest(state["source_manifest"])
    if manifest_path.exists():
        if _checked_manifest(json.loads(manifest_path.read_text())) != manifest:
            raise ValueError(
                "Baseline manifest disagrees with saved initialization; preserve this run "
                "and inspect its original state before recovering"
            )
    else:
        store.write_json(manifest_path, manifest)

    if not _archive_valid(archive_path, manifest):
        if store.source_manifest(state) != manifest:
            raise ValueError(
                "Source changed during incomplete baseline initialization; original files "
                "and partial archives were preserved. Restore files to "
                "baseline/source-manifest.json and resume, or start a new run directory"
            )
        if archive_path.exists():
            interrupted = folder / (
                "metax-source.interrupted-" + uuid.uuid4().hex + ".zip"
            )
            os.replace(archive_path, interrupted)
            _sync_directory(folder)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".metax-source-", suffix=".zip.partial", dir=folder
        )
        # Keep a failed temporary archive: it is evidence of interrupted capture.
        with os.fdopen(descriptor, "w+b") as stream:
            with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
                root = Path(state["inputs"]["metax_source"])
                for name, content_hash in sorted(manifest.items()):
                    if content_hash is not None:
                        archive.write(root / name, arcname=name)
            stream.flush()
            os.fsync(stream.fileno())
        if (
            not _archive_valid(temporary, manifest)
            or store.source_manifest(state) != manifest
        ):
            raise ValueError(
                "Source changed while capturing baseline; partial archive preserved. "
                "Restore frozen source files and resume, or start a new run directory"
            )
        os.replace(temporary, archive_path)
        _sync_directory(folder)

    state.setdefault("initialization", {})["phase"] = "environment"
    store.save(run_dir, state)


def runtime_identity(root=None):
    """Hash toolkit code and upgrade rules; timestamps never identify behavior."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[5]
    paths = set()
    skills = root / ".agents/skills"
    if skills.is_dir():
        for skill in skills.iterdir():
            if skill.name not in {
                "common-sense",
                "environment-check",
                "model-adaptation",
                "vllm-metax-upgrade",
            } and not skill.name.endswith("-upgrade"):
                continue
            paths.update(
                path for path in skill.rglob("*") if path.suffix in {".md", ".py"}
            )
    for entry in ("AGENTS.md", ".agents/agents/vllm-metax-upgrade.md"):
        path = root / entry
        if path.is_file():
            paths.add(path)
    files = {
        str(path.relative_to(root)): store.file_hash(path)
        for path in sorted(paths)
        if path.is_file() and "__pycache__" not in path.parts
    }
    return {"files": files, "fingerprint": store.digest(files)}


def _module_inputs(name, cwd):
    """Resolve ordinary files/packages without importing parents or running hooks."""
    if not name or any(not part.isidentifier() for part in name.split(".")):
        return []
    roots = list(dict.fromkeys([str(cwd), *(p for p in sys.path if p)]))
    suffixes = [".py", *importlib.machinery.EXTENSION_SUFFIXES]
    initializers = []
    for index, part in enumerate(name.split(".")):
        final = index == len(name.split(".")) - 1
        namespace = []
        found = None
        for base in roots:
            base = Path(base)
            package = base / part
            initializer = next(
                (
                    package / ("__init__" + suffix)
                    for suffix in suffixes
                    if (package / ("__init__" + suffix)).is_file()
                ),
                None,
            )
            if initializer is not None:
                initializers.append(initializer)
                found = [package]
                if final:
                    main = package / "__main__.py"
                    return [*initializers, *([main] if main.is_file() else [])]
                break
            if final:
                module = next(
                    (
                        base / (part + suffix)
                        for suffix in suffixes
                        if (base / (part + suffix)).is_file()
                    ),
                    None,
                )
                if module is not None:
                    return [*initializers, module]
            if package.is_dir():
                namespace.append(package)
        roots = found if found is not None else namespace
        if not roots:
            return initializers
        if final:
            main = next(
                (
                    root / "__main__.py"
                    for root in roots
                    if (root / "__main__.py").is_file()
                ),
                None,
            )
            return [*initializers, *([main] if main is not None else [])]
    return initializers


def evidence_inputs(command, cwd, extra_paths=()):
    """Identify direct command inputs; indirect imports/data need explicit paths.

    Shell expansion, stdin, custom import hooks and dependencies loaded by test
    runners are not inferred. ``extra_paths`` must name existing files, so a
    misspelled explicit input cannot silently disappear from evidence identity.
    """
    cwd = Path(cwd).resolve()
    command = [str(value) for value in command]
    files = set()
    identity = {}
    python = bool(command) and bool(
        re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", Path(command[0]).name)
        or Path(command[0]).resolve() == Path(sys.executable).resolve()
    )
    skip = set()
    if python:
        index = 1
        while index < len(command):
            value = command[index]
            if value == "-c" and index + 1 < len(command):
                identity["<python -c>"] = hashlib.sha256(
                    command[index + 1].encode()
                ).hexdigest()
                skip.update({index, index + 1})
                break
            if value == "-m" and index + 1 < len(command):
                files.update(_module_inputs(command[index + 1], cwd))
                skip.update({index, index + 1})
                break
            if not value.startswith("-"):
                break
            index += 2 if value in {"-W", "-X", "--check-hash-based-pycs"} else 1
    for index, argument in enumerate(command):
        if index in skip or (index == 0 and python):
            continue
        if index == 0:
            executable = Path(argument)
            if not executable.is_absolute():
                executable = (
                    cwd / executable
                    if "/" in argument
                    else Path(shutil.which(argument) or cwd / executable)
                )
            if executable.is_file():
                files.add(executable)
            continue
        candidate = (
            argument.split("=", 1)[1]
            if argument.startswith("-") and "=" in argument
            else argument
        )
        path = Path(candidate)
        path = path if path.is_absolute() else cwd / path
        if path.is_file():
            files.add(path)
    for name in extra_paths:
        path = Path(name)
        path = path if path.is_absolute() else cwd / path
        if not path.is_file():
            raise ValueError(
                "Explicit evidence input is not an existing file: " + str(path)
            )
        files.add(path)
    identity.update(
        {str(path.resolve()): store.file_hash(path) for path in sorted(files)}
    )
    return identity
