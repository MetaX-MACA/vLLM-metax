#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Create a read-only inventory of non-merge commits unique to a source ref."""

from __future__ import annotations

import argparse
import json
import os
import regex as re
import subprocess
import tempfile
from datetime import datetime, UTC
from pathlib import Path
from typing import Any
from collections.abc import Sequence


CHANGE_ID_RE = re.compile(r"^Change-Id:\s*(\S+)\s*$", re.MULTILINE | re.IGNORECASE)


class GitError(RuntimeError):
    """A Git command needed by the inventory failed."""


def _git(
    repo: Path,
    args: Sequence[str],
    *,
    input_bytes: bytes | None = None,
) -> bytes:
    command = ["git", "-C", str(repo), "-c", "core.quotepath=false", *args]
    result = subprocess.run(
        command,
        input=input_bytes,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise GitError(f"{' '.join(command)} failed: {detail}")
    return result.stdout


def _text(repo: Path, args: Sequence[str]) -> str:
    return _git(repo, args).decode("utf-8", errors="replace").strip()


def _resolve(repo: Path, ref: str) -> str:
    try:
        return _text(repo, ["rev-parse", "--verify", f"{ref}^{{commit}}"])
    except GitError as exc:
        raise GitError(f"cannot resolve {ref!r} as a commit: {exc}") from exc


def _commit_metadata(repo: Path, sha: str) -> dict[str, Any]:
    fmt = "%H%x00%P%x00%an%x00%ae%x00%aI%x00%cn%x00%ce%x00%cI%x00%B"
    fields = _text(repo, ["show", "-s", f"--format={fmt}", sha]).split("\0")
    if len(fields) != 9:
        raise GitError(f"unexpected metadata format for commit {sha}")

    message = fields[8].rstrip()
    subject = message.splitlines()[0] if message else ""
    change_ids = CHANGE_ID_RE.findall(message)
    return {
        "sha": fields[0],
        "parents": fields[1].split(),
        "author": {"name": fields[2], "email": fields[3], "date": fields[4]},
        "committer": {"name": fields[5], "email": fields[6], "date": fields[7]},
        "subject": subject,
        "message": message,
        "source_change_id": change_ids[-1] if change_ids else None,
    }


def _changed_files(repo: Path, sha: str) -> list[dict[str, Any]]:
    raw = _text(
        repo,
        ["diff-tree", "--root", "--no-commit-id", "--name-status", "-r", "-M", sha],
    )
    changes: list[dict[str, Any]] = []
    for line in raw.splitlines():
        fields = line.split("\t")
        if len(fields) < 2:
            continue
        changes.append({"status": fields[0], "paths": fields[1:]})
    return changes


def _patch_id(repo: Path, sha: str) -> str | None:
    patch = _git(repo, ["show", "--pretty=format:", "--no-ext-diff", "--binary", sha])
    output = _git(repo, ["patch-id", "--stable"], input_bytes=patch)
    line = output.decode("ascii", errors="replace").strip()
    return line.split()[0] if line else None


def _unique_commits(
    repo: Path, target_sha: str, source_sha: str, *, merges: bool
) -> list[str]:
    args = ["rev-list", "--reverse", "--topo-order"]
    args.append("--merges" if merges else "--no-merges")
    args.append(f"{target_sha}..{source_sha}")
    return [line for line in _text(repo, args).splitlines() if line]


def inventory(repo: Path, source: str, target: str) -> dict[str, Any]:
    """Return deterministic metadata for source-only non-merge commits."""
    repo = repo.resolve()
    source_sha = _resolve(repo, source)
    target_sha = _resolve(repo, target)
    if source_sha == target_sha:
        raise GitError("source and target resolve to the same commit")

    try:
        merge_base = _text(repo, ["merge-base", target_sha, source_sha])
    except GitError as exc:
        raise GitError("source and target do not share usable history") from exc

    commits = []
    for sha in _unique_commits(repo, target_sha, source_sha, merges=False):
        metadata = _commit_metadata(repo, sha)
        metadata["changed_files"] = _changed_files(repo, sha)
        metadata["patch_id"] = _patch_id(repo, sha)
        commits.append(metadata)

    ignored_merges = []
    for sha in _unique_commits(repo, target_sha, source_sha, merges=True):
        metadata = _commit_metadata(repo, sha)
        ignored_merges.append(
            {
                "sha": metadata["sha"],
                "parents": metadata["parents"],
                "author": metadata["author"],
                "subject": metadata["subject"],
                "source_change_id": metadata["source_change_id"],
            }
        )

    return {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "repository": str(repo),
        "source": {"ref": source, "sha": source_sha},
        "target": {"ref": target, "sha": target_sha},
        "merge_base": merge_base,
        "commit_count": len(commits),
        "ignored_merge_count": len(ignored_merges),
        "commits": commits,
        "ignored_merges": ignored_merges,
    }


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as temporary:
        temporary.write(payload)
        temporary_path = Path(temporary.name)
    os.replace(temporary_path, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inventory non-merge commits reachable from source but not target. "
            "The command is read-only; it never checks out or changes refs."
        )
    )
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--source", required=True, help="older source branch or ref")
    parser.add_argument("--target", required=True, help="newer target branch or ref")
    parser.add_argument("--output", type=Path, help="JSON path; stdout when omitted")
    args = parser.parse_args(argv)

    try:
        data = inventory(args.repo, args.source, args.target)
    except GitError as exc:
        parser.error(str(exc))

    if args.output:
        _write_json(args.output, data)
        print(args.output)
    else:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
