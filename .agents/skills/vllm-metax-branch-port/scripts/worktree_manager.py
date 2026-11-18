#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Manage run-scoped branch-port worktrees, leases and conservative cleanup."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import uuid
from collections.abc import Sequence
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import regex as re

SCHEMA_VERSION = 1
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
TERMINAL_STATUSES = {"PUBLISHED", "MERGED", "ABANDONED", "BLOCKED"}


def utc_now() -> datetime:
    return datetime.now(UTC)


def iso(value: datetime) -> str:
    return value.isoformat()


def parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value)


def git_output(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {repo}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def full_sha(repo: Path, ref: str) -> str:
    sha = git_output(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")
    if FULL_SHA_RE.fullmatch(sha) is None:
        raise ValueError(f"Resolved ref is not a full commit SHA: {ref}")
    return sha


def registry_path(repo: Path) -> Path:
    return repo / ".metax-branch-port" / "WORKTREES.json"


@contextmanager
def registry_lock(repo: Path):
    lock_path = repo / ".metax-branch-port" / "locks" / "worktrees.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def load_registry(repo: Path) -> dict[str, Any]:
    path = registry_path(repo)
    if not path.exists():
        return {"schema_version": SCHEMA_VERSION, "worktrees": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported WORKTREES.json schema version")
    if not isinstance(data.get("worktrees"), list):
        raise ValueError("WORKTREES.json must contain a worktrees list")
    return data


def save_registry(repo: Path, registry: dict[str, Any]) -> None:
    path = registry_path(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_run_worktree(run_dir: Path, record: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    temporary = run_dir / "WORKTREE.json.tmp"
    temporary.write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, run_dir / "WORKTREE.json")


def record_for_lease(records: list[dict[str, Any]], lease_id: str) -> dict[str, Any]:
    for record in records:
        if record.get("lease_id") == lease_id:
            return record
    raise ValueError(f"Unknown worktree lease: {lease_id}")


def active_lease(records: list[dict[str, Any]], edge_id: str) -> dict[str, Any] | None:
    now = utc_now()
    for record in records:
        if record.get("edge_id") != edge_id or record.get("status") != "ACTIVE":
            continue
        if parse_iso(record["lease_expires_at_utc"]) > now:
            return record
    return None


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    run_dir = args.run_dir.resolve()
    if SAFE_ID_RE.fullmatch(args.edge_id) is None:
        raise ValueError(
            "edge-id may contain only letters, digits, dot, underscore and hyphen"
        )
    source_sha = full_sha(repo, args.source_sha)
    target_sha = full_sha(repo, args.target_baseline_sha)
    worktree_root = (
        args.worktree_root.resolve()
        if args.worktree_root is not None
        else Path(
            os.environ.get(
                "METAX_BRANCH_PORT_WORKTREE_ROOT",
                str(repo.parent / ".metax-worktrees"),
            )
        ).resolve()
    )

    with registry_lock(repo):
        registry = load_registry(repo)
        records = registry["worktrees"]
        active = active_lease(records, args.edge_id)
        if active is not None:
            raise RuntimeError(
                "An active worktree lease already owns this edge: "
                f"{active['lease_id']} at {active['path']}"
            )

        run_id = (
            utc_now().strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + target_sha[:8]
            + "-"
            + uuid.uuid4().hex[:8]
        )
        path = worktree_root / args.edge_id / run_id
        if path.exists():
            raise RuntimeError(f"Run worktree path already exists: {path}")
        branch = f"port-run/{run_id}"
        path.parent.mkdir(parents=True, exist_ok=True)
        git_output(
            repo,
            "worktree",
            "add",
            "--no-track",
            "-b",
            branch,
            str(path),
            target_sha,
        )

        now = utc_now()
        lease_id = str(uuid.uuid4())
        record = {
            "run_id": run_id,
            "edge_id": args.edge_id,
            "path": str(path),
            "worktree_root": str(worktree_root),
            "local_branch": branch,
            "remote_pr_branch": args.remote_pr_branch,
            "source_sha": source_sha,
            "target_baseline_sha": target_sha,
            "head_sha": target_sha,
            "owner": args.owner,
            "lease_id": lease_id,
            "created_at_utc": iso(now),
            "last_verified_at_utc": iso(now),
            "lease_expires_at_utc": iso(now + timedelta(hours=args.lease_hours)),
            "status": "ACTIVE",
        }
        records.append(record)
        save_registry(repo, registry)
        write_run_worktree(run_dir, record)
        return record


def update_lease(args: argparse.Namespace, *, release: bool) -> dict[str, Any]:
    repo = args.repo.resolve()
    run_file = args.run_dir.resolve() / "WORKTREE.json"
    current = json.loads(run_file.read_text(encoding="utf-8"))
    lease_id = current["lease_id"]

    with registry_lock(repo):
        registry = load_registry(repo)
        record = record_for_lease(registry["worktrees"], lease_id)
        now = utc_now()
        record["last_verified_at_utc"] = iso(now)
        if release:
            if args.status not in TERMINAL_STATUSES:
                raise ValueError(
                    f"release status must be one of {sorted(TERMINAL_STATUSES)}"
                )
            record["status"] = args.status
            record["released_at_utc"] = iso(now)
            record["lease_expires_at_utc"] = iso(now)
        else:
            if record.get("status") != "ACTIVE":
                raise ValueError("Only ACTIVE worktrees can be heartbeated")
            record["lease_expires_at_utc"] = iso(
                now + timedelta(hours=args.lease_hours)
            )
        if args.head_sha is not None:
            record["head_sha"] = full_sha(repo, args.head_sha)
        save_registry(repo, registry)
        write_run_worktree(args.run_dir.resolve(), record)
        return record


def verify(args: argparse.Namespace) -> dict[str, Any]:
    data = json.loads(args.worktree_json.resolve().read_text(encoding="utf-8"))
    path = Path(data["path"]).resolve()
    problems: list[str] = []
    if not path.is_dir():
        problems.append(f"worktree path does not exist: {path}")
    else:
        actual_head = full_sha(path, "HEAD")
        actual_branch = git_output(path, "branch", "--show-current")
        status = git_output(path, "status", "--porcelain")
        if actual_head != data["head_sha"]:
            problems.append("worktree HEAD does not match recorded head_sha")
        if actual_branch != data["local_branch"]:
            problems.append("worktree branch does not match recorded local_branch")
        if status:
            problems.append("worktree is not clean")
    if data.get("status") != "ACTIVE":
        problems.append("worktree lease is not ACTIVE")
    if parse_iso(data["lease_expires_at_utc"]) <= utc_now():
        problems.append("worktree lease has expired")
    if problems:
        raise RuntimeError("; ".join(problems))
    return data


def cleanup(args: argparse.Namespace) -> list[dict[str, Any]]:
    repo = args.repo.resolve()
    cutoff = utc_now() - timedelta(days=args.older_than_days)
    removed: list[dict[str, Any]] = []
    with registry_lock(repo):
        registry = load_registry(repo)
        for record in registry["worktrees"]:
            if record.get("status") not in {"MERGED", "ABANDONED"}:
                continue
            if parse_iso(record["last_verified_at_utc"]) > cutoff:
                continue
            path = Path(record["path"]).resolve()
            worktree_root = Path(record.get("worktree_root", "")).resolve()
            if (
                not str(record.get("worktree_root", ""))
                or worktree_root not in path.parents
            ):
                continue
            if not path.is_dir():
                record["status"] = "REMOVED"
                continue
            if git_output(path, "status", "--porcelain"):
                continue
            head = full_sha(path, "HEAD")
            remote_contains = git_output(path, "branch", "-r", "--contains", head)
            if not remote_contains:
                continue
            if args.apply:
                git_output(repo, "worktree", "remove", str(path))
                record["status"] = "REMOVED"
                record["removed_at_utc"] = iso(utc_now())
            removed.append(record)
        if args.apply:
            save_registry(repo, registry)
            git_output(repo, "worktree", "prune")
    return removed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)

    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--repo", type=Path, required=True)
    prepare_parser.add_argument("--run-dir", type=Path, required=True)
    prepare_parser.add_argument("--edge-id", required=True)
    prepare_parser.add_argument("--source-sha", required=True)
    prepare_parser.add_argument("--target-baseline-sha", required=True)
    prepare_parser.add_argument("--remote-pr-branch")
    prepare_parser.add_argument(
        "--owner", default=os.environ.get("CODEX_THREAD_ID", "daily-agent")
    )
    prepare_parser.add_argument("--lease-hours", type=int, default=24)
    prepare_parser.add_argument("--worktree-root", type=Path)

    heartbeat_parser = commands.add_parser("heartbeat")
    heartbeat_parser.add_argument("--repo", type=Path, required=True)
    heartbeat_parser.add_argument("--run-dir", type=Path, required=True)
    heartbeat_parser.add_argument("--head-sha")
    heartbeat_parser.add_argument("--lease-hours", type=int, default=24)

    release_parser = commands.add_parser("release")
    release_parser.add_argument("--repo", type=Path, required=True)
    release_parser.add_argument("--run-dir", type=Path, required=True)
    release_parser.add_argument("--status", required=True)
    release_parser.add_argument("--head-sha")

    verify_parser = commands.add_parser("verify")
    verify_parser.add_argument("--worktree-json", type=Path, required=True)

    list_parser = commands.add_parser("list")
    list_parser.add_argument("--repo", type=Path, required=True)

    cleanup_parser = commands.add_parser("cleanup")
    cleanup_parser.add_argument("--repo", type=Path, required=True)
    cleanup_parser.add_argument("--older-than-days", type=int, default=14)
    cleanup_parser.add_argument("--apply", action="store_true")

    args = parser.parse_args(argv)
    try:
        if args.action == "prepare":
            result = prepare(args)
        elif args.action == "heartbeat":
            result = update_lease(args, release=False)
        elif args.action == "release":
            result = update_lease(args, release=True)
        elif args.action == "verify":
            result = verify(args)
        elif args.action == "list":
            result = load_registry(args.repo.resolve())
        else:
            result = cleanup(args)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print(f"worktree manager blocked: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
