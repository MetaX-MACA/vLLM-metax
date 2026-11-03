#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Sync the version pins of requirements/maca_private.txt with the build env.
The pins of maca_private.txt point at the MetaX/MACA index and carry build
specific local versions, e.g.
    mcoplib==0.4.13+gd047a40.maca3.8.2.6.torch2.10
    nixl==1.0.0+maca3.8.2.302
They drift: the container may hold another local version, or the pipeline may
build one in this very run. Since setup.py turns this file into
install_requires, a stale pin ends up in the metadata of the built wheel.
This tool rewrites the version part of every pinned line with the version found
in the build environment, in the given wheel directories, or with an explicit
--set override. The NIXL and MIXL distribution names are aliases during the
rename transition, and the project name of the MIXL source checkout takes
precedence when --scan-root is supplied. Comments, blank lines, option lines
and unpinned lines are kept as they are.
Examples:
    python tools/sync_maca_private.py                    # report only
    python tools/sync_maca_private.py --write            # rewrite in place
    python tools/sync_maca_private.py --check            # exit 1 if out of sync
    python tools/sync_maca_private.py --wheel-dir out/wheel --wheel-dir /opt/maca/wheel
    python tools/sync_maca_private.py --set nixl=1.0.0+maca3.8.2.302
"""

import argparse
import importlib.metadata as importlib_metadata
import regex as re
import sys
from pathlib import Path
from packaging.version import InvalidVersion, Version

DEFAULT_FILE = Path("requirements/maca_private.txt")
NIXL_DIST_NAMES = ("nixl", "mixl")
PROJECT_NAME_RE = re.compile(
    r'^\s*name\s*=\s*["\'](?P<name>[^"\']+)["\']', re.MULTILINE
)
# name==version, optionally followed by an inline comment
PIN_RE = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)\s*==\s*(?P<version>[^\s#]+)\s*$"
)


def canonical(name: str) -> str:
    """PEP 503 name normalization, so flash_attn_3 == flash-attn-3 == flash.attn.3."""
    return re.sub(r"[-_.]+", "-", name).lower()


def nixl_distribution_from_sources(roots):
    """Return the nixl/mixl distribution that the checked-out MIXL project builds."""
    for root in roots:
        root = Path(root)
        for source_dir in (root, root / "mixl", root / "nixl"):
            for filename in ("pyproject.toml", "pyproject.toml.in"):
                project_file = source_dir / filename
                if not project_file.is_file():
                    continue
                match = PROJECT_NAME_RE.search(project_file.read_text())
                if match:
                    name = canonical(match.group("name"))
                    if name in NIXL_DIST_NAMES:
                        return name
    return None


def versions_from_environment() -> dict:
    """Versions of the distributions installed in the current (build) environment."""
    found = {}
    for dist in importlib_metadata.distributions():
        try:
            name = dist.metadata["Name"]
            version = dist.version
        except Exception:  # broken metadata must not break the build
            continue
        if name and version:
            found.setdefault(canonical(name), version)
    return found


def wheel_name_version(wheel: Path):
    """("nixl", "1.0.0+maca3.8.2.302") from nixl-1.0.0+maca3.8.2.302-cp310-...whl."""
    parts = wheel.name[: -len(".whl")].split("-")
    if len(parts) < 5:  # not name-version-python-abi-platform
        return None
    return parts[0], parts[1]


def versions_from_wheels(dirs) -> dict:
    """Versions parsed from the wheel file names of the given directories."""
    found = {}
    for directory in dirs:
        for wheel in sorted(Path(directory).glob("*.whl")):
            parsed = wheel_name_version(wheel)
            if parsed:
                found.setdefault(canonical(parsed[0]), parsed[1])
    return found


def is_newer_version(candidate: str, current: str) -> bool:
    """Compare PEP 440 versions, falling back to lexical order if needed."""
    try:
        return Version(candidate) > Version(current)
    except InvalidVersion:
        return candidate > current


def versions_from_scan(roots) -> dict:
    """Same, but walking the wheel layouts used by the build pipeline.
    The pipeline extracts every package it ships into the workspace with that
    layout (<root>/*/wheel, <root>/*/*/wheel, and torch-specific wheel
    variants such as <root>/*/wheel/torch2.10 and
    <root>/*/*/wheel/torch2.10), so a scan picks up all of them instead of a
    hand written list of directories.
    """
    found = {}
    dirs = []
    for root in roots:
        dirs += sorted(str(p) for p in Path(root).glob("*/wheel"))
        dirs += sorted(str(p) for p in Path(root).glob("*/*/wheel"))
        dirs += sorted(str(p) for p in Path(root).glob("*/wheel/*"))
        dirs += sorted(str(p) for p in Path(root).glob("*/*/wheel/*"))
    for directory in dirs:
        for name, version in versions_from_wheels([directory]).items():
            current = found.get(name)
            if current is None or is_newer_version(version, current):
                found[name] = version
    return found


def resolve(args) -> dict:
    # lower precedence first, so that more specific sources win
    resolved = versions_from_environment()
    for name, version in versions_from_scan(args.scan_root).items():
        resolved[name] = version
    for name, version in versions_from_wheels(args.wheel_dir).items():
        resolved[name] = version
    for item in args.set:
        name, sep, version = item.partition("=")
        if not sep or not name or not version:
            raise SystemExit(f"error: --set expects NAME=VERSION, got {item!r}")
        name = canonical(name)
        if name in NIXL_DIST_NAMES:
            # The build script historically used --set nixl for both names.
            for alias in NIXL_DIST_NAMES:
                resolved[alias] = version
        else:
            resolved[name] = version
    return resolved


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("file", nargs="?", default=str(DEFAULT_FILE))
    parser.add_argument(
        "--wheel-dir",
        action="append",
        default=[],
        metavar="DIR",
        help="directory with *.whl to take versions from (repeatable)",
    )
    parser.add_argument(
        "--scan-root",
        action="append",
        default=[],
        metavar="DIR",
        help="workspace root to scan for */wheel and */*/wheel directories (repeatable)",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VERSION",
        help="force a version, e.g. for a wheel built later in the pipeline",
    )
    parser.add_argument(
        "--write", action="store_true", help="rewrite the file in place"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="do not write, exit with 1 when the file is out of sync",
    )
    args = parser.parse_args(argv)
    path = Path(args.file)
    if not path.is_file():
        print(f"error: {path} not found", file=sys.stderr)
        return 2
    resolved = resolve(args)
    source_nixl_dist = nixl_distribution_from_sources(args.scan_root)
    updated, in_sync, unresolved, skipped = [], [], [], []
    lines = []
    for line in path.read_text().splitlines(keepends=True):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("--"):
            lines.append(line)
            continue
        core, newline = (line[:-1], "\n") if line.endswith("\n") else (line, "")
        body, sep, comment = core.partition("#")
        match = PIN_RE.match(body.rstrip())
        if not match:
            skipped.append(stripped)
            lines.append(line)
            continue
        name, old_version = match.group("name"), match.group("version")
        new_name = name
        if canonical(name) in NIXL_DIST_NAMES and source_nixl_dist:
            new_name = source_nixl_dist
        new_version = resolved.get(canonical(name))
        if new_version is None:
            unresolved.append(f"{name}=={old_version}")
            lines.append(line)
            continue
        if new_name == name and new_version == old_version:
            in_sync.append(f"{name}=={old_version}")
            lines.append(line)
            continue
        trailing = body[len(body.rstrip()) :]
        updated.append(f"{name}=={old_version} -> {new_name}=={new_version}")
        lines.append(f"{new_name}=={new_version}{trailing}{sep}{comment}{newline}")
    for item in updated:
        print(f"update {item}")
    for item in unresolved:
        print(f"unresolved (kept as is): {item}", file=sys.stderr)
    for item in skipped:
        print(f"not a '== pin' (kept as is): {item}", file=sys.stderr)
    print(
        f"{path}: {len(updated)} updated, {len(in_sync)} in sync, "
        f"{len(unresolved)} unresolved, {len(skipped)} skipped"
    )
    if updated and args.write and not args.check:
        path.write_text("".join(lines))
        print(f"wrote {path}")
    if updated and args.check:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
