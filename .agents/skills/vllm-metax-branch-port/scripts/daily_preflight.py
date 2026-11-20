#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Validate all unattended daily branch-port prerequisites before mutation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import regex as re

from agent_paths import config_path, registry_path
from daily_config import git_output, validate_config
from publish_branch_port import inspect_publication
from runtime_fingerprint import compute_fingerprint

FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
IDENTITY_RE = re.compile(r"^.+ <([^<>@\s]+@[^<>\s]+)>$")


def now() -> str:
    return datetime.now(UTC).isoformat()


def current_authors(repo: Path, source_sha: str, target_sha: str) -> list[str]:
    shas = git_output(repo, "rev-list", "--no-merges", f"{target_sha}..{source_sha}")
    return sorted(
        {
            git_output(repo, "show", "-s", "--format=%an <%ae>", sha)
            for sha in shas.splitlines()
            if sha
        }
    )


def active_lease(repo: Path, edge_id: str) -> dict[str, Any] | None:
    path = registry_path(repo)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    current = datetime.now(UTC)
    for record in data.get("worktrees", []):
        if record.get("edge_id") != edge_id or record.get("status") != "ACTIVE":
            continue
        expires = datetime.fromisoformat(record["lease_expires_at_utc"])
        if expires > current:
            return record
    return None


def check_remote_access(repo: Path, remote: str, branch: str) -> str | None:
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "ls-remote",
            "--heads",
            remote,
            f"refs/heads/{branch}",
        ],
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    if result.returncode:
        return result.stderr.strip() or f"git ls-remote failed for {remote}/{branch}"
    return None


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    repo = args.repo.resolve()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    validate_config(repo, config)
    checks: list[dict[str, str]] = []
    warnings: list[str] = []
    blockers: list[str] = []

    for name, value in (
        ("source_sha", args.source_sha),
        ("target_sha", args.target_sha),
        ("master_sha", args.master_sha),
    ):
        if FULL_SHA_RE.fullmatch(value) is None:
            blockers.append(f"{name} must be a full SHA")

    developer = config["identity"]["porting_developer"]
    identity_match = IDENTITY_RE.fullmatch(developer)
    if identity_match is None:
        blockers.append("identity.porting_developer is malformed")
    else:
        expected_email = identity_match.group(1)
        git_name = git_output(repo, "config", "user.name")
        git_email = git_output(repo, "config", "user.email")
        if not git_name or not git_email:
            blockers.append("Git user.name/user.email are not configured")
        elif f"{git_name} <{git_email}>" != developer:
            warnings.append(
                "repository Git identity differs; commits must explicitly use the approved identity"
            )
        checks.append(
            {
                "name": "git_identity",
                "status": "PASSED",
                "evidence": f"approved identity {developer}; expected email {expected_email}",
            }
        )

    authors = current_authors(repo, args.source_sha, args.target_sha)
    approved_authors = set(config["scope"]["approved_source_authors"])
    unknown_authors = sorted(set(authors) - approved_authors)
    if unknown_authors and config["scope"]["unknown_source_author"] == "block":
        blockers.append("unknown source authors: " + ", ".join(unknown_authors))
    elif unknown_authors:
        warnings.append(
            "unknown source authors allowed by policy: " + ", ".join(unknown_authors)
        )
    checks.append(
        {
            "name": "source_authors",
            "status": "PASSED" if not unknown_authors else "WARN",
            "evidence": f"{len(authors)} authors; {len(unknown_authors)} unknown",
        }
    )

    fingerprint = compute_fingerprint()
    if fingerprint["fingerprint"] != config["environment"]["fingerprint"]:
        message = (
            "environment fingerprint changed: "
            f"approved={config['environment']['fingerprint']} current={fingerprint['fingerprint']}"
        )
        if config["environment"]["environment_mismatch"] == "block":
            blockers.append(message)
        else:
            warnings.append(message + "; environment re-verification is required")
    checks.append(
        {
            "name": "environment_fingerprint",
            "status": "PASSED"
            if fingerprint["fingerprint"] == config["environment"]["fingerprint"]
            else "WARN",
            "evidence": fingerprint["fingerprint"],
        }
    )

    kernel_value = os.environ.get("USE_PRECOMPILED_KERNEL")
    kernel_provider = (
        "precompiled"
        if kernel_value in {"1", "true", "True"}
        else "source"
        if kernel_value in {"0", "false", "False", None}
        else "unknown"
    )
    allowed_provider = config["environment"]["allowed_kernel_provider"]
    if kernel_provider == "unknown" or (
        allowed_provider != "either" and kernel_provider != allowed_provider
    ):
        blockers.append(
            f"kernel provider {kernel_provider} does not match {allowed_provider}"
        )

    if args.reviewer_capability != "available":
        blockers.append(
            "fresh independent reviewer capability is not confirmed as available"
        )

    publication = config["publication"]
    approvals = config["approvals"]
    if not approvals["remote_publication_approved"]:
        blockers.append("remote publication is not approved")
    if (
        publication["publish_with_pending_review"]
        and not approvals["draft_publication_with_pending_approved"]
    ):
        blockers.append("pending-review draft publication is not approved")

    report = inspect_publication(repo, config["source_ref"], config["target_ref"])
    if report.get("status") != "ok":
        blockers.extend(report.get("issues", ["publication inspection failed"]))
    else:
        target_provider = report["target"]["provider"]
        target_remote = report["target"]["remote"]
        if target_provider == "github":
            if shutil.which("gh") is None:
                blockers.append("gh CLI is not installed")
            else:
                auth = subprocess.run(
                    ["gh", "auth", "status"],
                    text=True,
                    capture_output=True,
                    check=False,
                )
                if auth.returncode:
                    blockers.append("gh CLI authentication is unavailable")
        if target_remote:
            error = check_remote_access(repo, target_remote, config["target_ref"])
            if error:
                blockers.append(f"target remote access failed: {error}")
        source_remote = report["source"]["remote"]
        if source_remote:
            error = check_remote_access(repo, source_remote, config["source_ref"])
            if error:
                blockers.append(f"source remote access failed: {error}")
        checks.append(
            {
                "name": "publication_inspection",
                "status": "PASSED",
                "evidence": (
                    f"source={report['source']['provider']} "
                    f"target={report['target']['provider']}"
                ),
            }
        )

    root = Path(config["worktree"]["root"])
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / f".preflight-{os.getpid()}"
        probe.write_text("ok\n")
        probe.unlink()
        checks.append(
            {"name": "worktree_root", "status": "PASSED", "evidence": str(root)}
        )
    except OSError as error:
        blockers.append(f"worktree root is not writable: {error}")

    edge_id = (
        config["source_ref"].replace("/", "-")
        + "-to-"
        + config["target_ref"].replace("/", "-")
    )
    lease = active_lease(repo, edge_id)
    lease_status = "none"
    if lease:
        lease_status = "active"
        if config["worktree"]["active_lease_policy"] == "block":
            blockers.append(f"active worktree lease exists: {lease['lease_id']}")
        else:
            warnings.append(
                "active worktree lease exists; wait up to "
                f"{config['worktree']['active_lease_wait_minutes']} minutes"
            )

    status = "BLOCKED" if blockers else ("WAIT" if lease else "READY")
    return {
        "status": status,
        "checked_at_utc": now(),
        "source_sha": args.source_sha,
        "target_sha": args.target_sha,
        "master_sha": args.master_sha,
        "edge_id": edge_id,
        "blockers": blockers,
        "warnings": warnings,
        "checks": checks,
        "active_lease": lease,
        "lease_status": lease_status,
        "environment": fingerprint,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--target-sha", required=True)
    parser.add_argument("--master-sha", required=True)
    parser.add_argument(
        "--reviewer-capability",
        choices=("available", "unavailable", "unknown"),
        default="unknown",
    )
    args = parser.parse_args(argv)
    args.config = args.config or config_path(args.repo.resolve())
    try:
        result = preflight(args)
    except (OSError, ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
        print(f"daily preflight blocked: {error}")
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] in {"READY", "WAIT"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
