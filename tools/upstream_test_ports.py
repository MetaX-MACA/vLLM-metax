# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Inspect and validate the file-by-file upstream test port ledger."""

import argparse
import subprocess
import tomllib
from collections import Counter
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "tests" / "upstream_test_ports.toml"
STATUSES = {"partial", "ported", "not_applicable"}
TEST_ROOT = PurePosixPath("tests")


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise ValueError(result.stderr.strip())
    return result.stdout


def resolve(repo: Path, revision: str) -> str:
    return git(
        repo, "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"
    ).strip()


def is_test(path: str) -> bool:
    file = PurePosixPath(path)
    return file.suffix == ".py" and (
        file.stem.startswith("test_") or file.stem.endswith("_test")
    )


def test_inventory(repo: Path, revision: str) -> dict[str, str]:
    inventory = {}
    for entry in git(repo, "ls-tree", "-rz", revision, "--", "tests").split("\0"):
        if entry:
            metadata, path = entry.split("\t", 1)
            _, kind, blob = metadata.split()
            if kind == "blob" and is_test(path):
                inventory[path] = blob
    return inventory


def validate(ports: dict, inventory: dict[str, str]) -> list[str]:
    errors = []
    for path, entry in ports.items():
        status = entry.get("status")
        if status not in STATUSES:
            errors.append(f"{path}: unknown status {status!r}")
        if path not in inventory:
            errors.append(f"{path}: missing from the upstream test inventory")
        elif entry.get("blob") != inventory[path]:
            errors.append(
                f"{path}: upstream blob changed; review the file before updating the ledger"
            )
        if not entry.get("notes", "").strip():
            errors.append(f"{path}: missing applicability or port notes")
        local_test_exists = (ROOT / path).is_file()
        if status in {"ported", "partial"}:
            if not local_test_exists:
                errors.append(f"{path}: missing corresponding plugin test file")
            if not entry.get("implementation"):
                errors.append(f"{path}: missing plugin implementation")
        elif status == "not_applicable" and local_test_exists:
            errors.append(f"{path}: local test contradicts the not_applicable status")
        for implementation in entry.get("implementation", []):
            implementation_path = PurePosixPath(implementation)
            if (
                implementation_path.parts[:1] not in {("vllm_metax",), ("csrc",)}
                or ".." in implementation_path.parts
                or not (ROOT / implementation).is_file()
            ):
                errors.append(f"{path}: invalid plugin implementation {implementation}")
    for path in sorted(inventory.keys() - ports.keys()):
        if (ROOT / path).is_file():
            errors.append(f"{path}: local test has no ledger entry")
    return errors


def list_inventory(
    inventory: dict[str, str], ports: dict, prefix: PurePosixPath, summary: bool
) -> None:
    counts = Counter()
    for path, blob in sorted(inventory.items()):
        if PurePosixPath(path).is_relative_to(prefix):
            status = ports.get(path, {}).get("status", "pending")
            counts[status] += 1
            if not summary:
                print(f"{status}\t{blob}\t{path}")
    print(" ".join(f"{key}={counts[key]}" for key in sorted(STATUSES | {"pending"})))


def list_changes(
    repo: Path, baseline: str, target: str, ports: dict, prefix: PurePosixPath
) -> None:
    changes = iter(
        git(
            repo,
            "diff",
            "--name-status",
            "-z",
            "--find-renames",
            baseline,
            target,
            "--",
            "tests",
        )
        .rstrip("\0")
        .split("\0")
    )
    for change in changes:
        if not change:
            continue
        paths = [next(changes)]
        if change.startswith(("R", "C")):
            paths.append(next(changes))
        files = [PurePosixPath(path) for path in paths]
        kind = "test" if any(is_test(path) for path in paths) else "support"
        in_prefix = any(file.is_relative_to(prefix) for file in files)
        parent_conftest = any(
            file.name == "conftest.py" and prefix.is_relative_to(file.parent)
            for file in files
        )
        if in_prefix or parent_conftest:
            if kind == "support":
                review = "review_dependents"
            else:
                review = "review_again" if paths[0] in ports else "pending"
            print("\t".join((change, kind, review, *paths)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--upstream", required=True, type=Path, help="local upstream vLLM checkout"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "check", help="validate the ledger at its pinned upstream revision"
    )
    inventory_parser = commands.add_parser(
        "inventory", help="list upstream tests and port status"
    )
    inventory_parser.add_argument("--prefix", default=TEST_ROOT, type=PurePosixPath)
    inventory_parser.add_argument("--summary", action="store_true")
    changes_parser = commands.add_parser(
        "changes", help="report test and fixture changes for a new upstream revision"
    )
    changes_parser.add_argument("target")
    changes_parser.add_argument("--prefix", default=TEST_ROOT, type=PurePosixPath)
    args = parser.parse_args()

    try:
        with LEDGER.open("rb") as file:
            ledger = tomllib.load(file)
        baseline = resolve(args.upstream, ledger["revision"])
        if baseline != ledger["revision"]:
            raise ValueError("ledger revision must be a full upstream commit ID")
        ports = ledger.get("files", {})
        inventory = test_inventory(args.upstream, baseline)
        errors = validate(ports, inventory)
        if errors:
            parser.exit(1, "\n".join(errors) + "\n")
        if args.command == "check":
            print(f"Valid ledger for {baseline}")
        elif args.command == "inventory":
            list_inventory(inventory, ports, args.prefix, args.summary)
        else:
            target = resolve(args.upstream, args.target)
            list_changes(args.upstream, baseline, target, ports, args.prefix)
    except (KeyError, OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
