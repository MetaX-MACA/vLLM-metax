#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Compute and record idempotent daily branch-port run keys."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import regex as re

from agent_paths import policy_root, state_path

FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
POLICY_PATHS = (
    "AGENTS.md",
    ".agents/agents/vllm-metax-branch-port.md",
    ".agents/agents/vllm-metax-branch-port-reviewer.md",
    ".agents/agents/vllm-metax-branch-port-daily.md",
    ".agents/skills/vllm-metax-branch-port",
)
SKIP_PARTS = {"__pycache__", ".pytest_cache"}


def now() -> str:
    return datetime.now(UTC).isoformat()


def require_sha(value: str, name: str) -> str:
    if FULL_SHA_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a full 40-character SHA")
    return value


def file_digest(path: Path, digest: Any) -> None:
    if path.is_file():
        digest.update(path.read_bytes())
        return
    if not path.is_dir():
        raise ValueError(f"Policy path does not exist: {path}")
    for child in sorted(path.rglob("*")):
        if not child.is_file() or SKIP_PARTS.intersection(child.parts):
            continue
        digest.update(str(child.relative_to(path)).encode())
        digest.update(child.read_bytes())


def compute_run_key(
    repo: Path,
    *,
    config: Path,
    intent_file: Path | None,
    source_sha: str,
    target_sha: str,
    master_sha: str,
    environment_fingerprint: str,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    fields = {
        "source_sha": require_sha(source_sha, "source_sha"),
        "target_sha": require_sha(target_sha, "target_sha"),
        "master_sha": require_sha(master_sha, "master_sha"),
        "environment_fingerprint": environment_fingerprint,
    }
    digest.update(json.dumps(fields, sort_keys=True).encode())
    for relative in POLICY_PATHS:
        digest.update(relative.encode())
        file_digest(policy_root(repo) / relative, digest)
    digest.update(config.read_bytes())
    if intent_file is not None:
        digest.update(intent_file.read_bytes())
        fields["intent_file"] = str(intent_file.resolve())
    fields["run_key"] = digest.hexdigest()
    return fields


def load_state(repo: Path) -> dict[str, Any]:
    path = state_path(repo)
    if not path.exists():
        return {"schema_version": 1, "last_completed_run_key": None, "runs": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(repo: Path, state: dict[str, Any]) -> None:
    path = state_path(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--intent-file", type=Path)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--target-sha", required=True)
    parser.add_argument("--master-sha", required=True)
    parser.add_argument("--environment-fingerprint", required=True)
    parser.add_argument("--record-status", choices=("COMPLETED", "BLOCKED", "NOOP"))
    parser.add_argument("--analysis-status", choices=("CHANGED", "UNCHANGED"))
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--target-head-sha")
    parser.add_argument("--artifact", action="append", default=[])
    args = parser.parse_args(argv)

    try:
        repo = args.repo.resolve()
        result = compute_run_key(
            repo,
            config=args.config.resolve(),
            intent_file=args.intent_file.resolve() if args.intent_file else None,
            source_sha=args.source_sha,
            target_sha=args.target_sha,
            master_sha=args.master_sha,
            environment_fingerprint=args.environment_fingerprint,
        )
        state = load_state(repo)
        previous = state.get("last_completed_run_key")
        result["previous_completed_run_key"] = previous
        result["noop"] = previous == result["run_key"]

        if args.record_status:
            if (
                args.record_status in {"COMPLETED", "NOOP"}
                and args.analysis_status != "UNCHANGED"
            ):
                raise ValueError(
                    f"{args.record_status} requires --analysis-status UNCHANGED"
                )
            if args.record_status == "BLOCKED" and args.analysis_status != "CHANGED":
                raise ValueError("BLOCKED requires --analysis-status CHANGED")
            run = {
                **result,
                "status": args.record_status,
                "recorded_at_utc": now(),
                "run_dir": str(args.run_dir.resolve()) if args.run_dir else None,
                "target_head_sha": (
                    require_sha(args.target_head_sha, "target_head_sha")
                    if args.target_head_sha
                    else None
                ),
                "artifacts": [str(Path(item).resolve()) for item in args.artifact],
            }
            state.setdefault("runs", []).append(run)
            if args.record_status in {"COMPLETED", "NOOP"}:
                state["last_completed_run_key"] = result["run_key"]
            save_state(repo, state)
            result["recorded"] = run

        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"daily state blocked: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
