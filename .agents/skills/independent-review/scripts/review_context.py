#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Create or verify an exact code-review context for a Git target."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MODES = {"staged", "worktree", "commit", "range"}


def now() -> str:
    return datetime.now(UTC).isoformat()


def git_bytes(repo: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        check=False,
    )
    if result.returncode:
        stderr = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"git {' '.join(args)} failed in {repo}: {stderr}")
    return result.stdout


def git_text(repo: Path, *args: str) -> str:
    return git_bytes(repo, *args).decode().strip()


def full_sha(repo: Path, ref: str) -> str:
    return git_text(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")


def untracked_manifest(repo: Path) -> tuple[list[str], str]:
    output = git_bytes(repo, "ls-files", "--others", "--exclude-standard", "-z")
    paths = sorted(path.decode() for path in output.split(b"\0") if path)
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.encode())
        digest.update((repo / path).read_bytes())
    return paths, digest.hexdigest()


def changed_files(repo: Path, *args: str) -> list[str]:
    output = git_text(repo, "diff", "--name-status", *args)
    return [line for line in output.splitlines() if line]


def build_context(
    repo: Path,
    *,
    mode: str,
    base: str | None = None,
    head: str | None = None,
) -> dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {sorted(MODES)}")
    repo = repo.resolve()
    branch = git_text(repo, "branch", "--show-current")
    index_tree_sha: str | None = None
    untracked_files: list[str] = []
    untracked_sha256 = hashlib.sha256().hexdigest()

    if mode == "staged":
        current = full_sha(repo, "HEAD")
        base_sha = current
        head_sha = current
        diff = git_bytes(repo, "diff", "--cached", "--binary")
        index_tree_sha = git_text(repo, "write-tree")
        files = changed_files(repo, "--cached")
    elif mode == "worktree":
        current = full_sha(repo, "HEAD")
        base_sha = current
        head_sha = current
        diff = git_bytes(repo, "diff", "--binary")
        files = changed_files(repo)
        untracked_files, untracked_sha256 = untracked_manifest(repo)
        files.extend(f"??\t{path}" for path in untracked_files)
    elif mode == "commit":
        base_ref = base or "HEAD^"
        head_ref = head or "HEAD"
        base_sha = full_sha(repo, base_ref)
        head_sha = full_sha(repo, head_ref)
        diff = git_bytes(repo, "diff", "--binary", base_sha, head_sha)
        files = changed_files(repo, base_sha, head_sha)
    else:
        if not base or not head:
            raise ValueError("range mode requires --base and --head")
        base_sha = full_sha(repo, base)
        head_sha = full_sha(repo, head)
        diff = git_bytes(repo, "diff", "--binary", base_sha, head_sha)
        files = changed_files(repo, base_sha, head_sha)

    review_payload = {
        "review_mode": mode,
        "base_sha": base_sha,
        "head_sha": head_sha,
        "index_tree_sha": index_tree_sha,
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked_sha256": untracked_sha256,
    }
    review_hash = hashlib.sha256(
        json.dumps(review_payload, sort_keys=True).encode()
    ).hexdigest()
    return {
        "schema_version": 1,
        "review_mode": mode,
        **review_payload,
        "repo": str(repo),
        "worktree": str(repo),
        "branch": branch,
        "review_hash": review_hash,
        "changed_files": files,
        "untracked_files": untracked_files,
        "created_at_utc": now(),
    }


def verify_context(context: dict[str, Any]) -> dict[str, Any]:
    current = build_context(
        Path(context["repo"]),
        mode=context["review_mode"],
        base=context["base_sha"]
        if context["review_mode"] in {"commit", "range"}
        else None,
        head=context["head_sha"]
        if context["review_mode"] in {"commit", "range"}
        else None,
    )
    keys = (
        "review_mode",
        "base_sha",
        "head_sha",
        "index_tree_sha",
        "review_hash",
        "changed_files",
        "untracked_files",
    )
    mismatches = [key for key in keys if context.get(key) != current.get(key)]
    if mismatches:
        raise ValueError("review context is stale: " + ", ".join(mismatches))
    return current


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    snapshot = commands.add_parser("snapshot")
    snapshot.add_argument("--repo", type=Path, required=True)
    snapshot.add_argument("--mode", choices=sorted(MODES), required=True)
    snapshot.add_argument("--base")
    snapshot.add_argument("--head")
    snapshot.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--context", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        if args.action == "snapshot":
            result = build_context(
                args.repo, mode=args.mode, base=args.base, head=args.head
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.output.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(result, indent=2, ensure_ascii=False) + "\n"
            )
            os.replace(temporary, args.output)
        else:
            context = json.loads(args.context.read_text(encoding="utf-8"))
            result = verify_context(context)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"review context blocked: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
