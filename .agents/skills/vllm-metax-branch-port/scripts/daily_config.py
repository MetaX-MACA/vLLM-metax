#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Detect and create durable daily branch-port configuration."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import regex as re

from runtime_fingerprint import compute_fingerprint

MX_BRANCH_RE = re.compile(r"^mx/v(?P<version>\d+\.\d+\.\d+)-dev$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")


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


def config_path(repo: Path) -> Path:
    return repo / ".metax-branch-port" / "DAILY-CONFIG.json"


def scope_path(repo: Path) -> Path:
    return repo / ".metax-branch-port" / "DAILY-SCOPE.json"


def atomic_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def remote_names(repo: Path) -> list[str]:
    return [line for line in git_output(repo, "remote").splitlines() if line]


def remote_urls(repo: Path, remote: str) -> list[str]:
    output = git_output(repo, "remote", "get-url", "--all", remote)
    return [line for line in output.splitlines() if line]


def remote_branches(repo: Path) -> list[str]:
    output = git_output(
        repo,
        "for-each-ref",
        "--format=%(refname:short)",
        "refs/remotes",
    )
    return [line for line in output.splitlines() if line]


def branch_candidates(repo: Path) -> dict[str, Any]:
    branches = remote_branches(repo)
    mx_versions: list[tuple[tuple[int, int, int], str]] = []
    for full in branches:
        branch = full.split("/", 1)[1] if "/" in full else full
        match = MX_BRANCH_RE.fullmatch(branch)
        if match:
            version = tuple(int(part) for part in match.group("version").split("."))
            mx_versions.append((version, branch))
    mx_versions.sort(reverse=True)
    remotes = remote_names(repo)
    github_remotes = [
        remote
        for remote in remotes
        if any(
            re.search(r"github\.com[/:]", url, re.IGNORECASE)
            for url in remote_urls(repo, remote)
        )
    ]
    master_available = any(
        branch == "master" or branch.endswith("/master") for branch in branches
    )
    return {
        "current_branch": git_output(repo, "branch", "--show-current"),
        "latest_mx_dev": mx_versions[0][1] if mx_versions else None,
        "mx_dev_branches": [branch for _, branch in mx_versions],
        "master_available": master_available,
        "remotes": remotes,
        "github_remotes": github_remotes,
    }


def resolve_ref(repo: Path, ref: str) -> str:
    for candidate in (ref, f"origin/{ref}", f"upstream/{ref}"):
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "rev-parse",
                "--verify",
                f"{candidate}^{{commit}}",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    raise ValueError(f"Cannot resolve Git ref: {ref}")


def source_authors(repo: Path, source_ref: str, target_ref: str) -> list[str]:
    source_sha = resolve_ref(repo, source_ref)
    target_sha = resolve_ref(repo, target_ref)
    shas = git_output(repo, "rev-list", "--no-merges", f"{target_sha}..{source_sha}")
    authors = set()
    for sha in shas.splitlines():
        if sha:
            authors.add(git_output(repo, "show", "-s", "--format=%an <%ae>", sha))
    return sorted(authors)


def validate_config(repo: Path, config: dict[str, Any]) -> dict[str, Any]:
    problems: list[str] = []
    if config.get("schema_version") != 1:
        problems.append("schema_version must be 1")
    for field in ("source_ref", "target_ref"):
        if not isinstance(config.get(field), str) or not config[field].strip():
            problems.append(f"{field} must be non-empty")

    scope = config.get("scope")
    if not isinstance(scope, dict):
        problems.append("scope must be an object")
    elif scope.get("mode") not in {"all-source-only", "explicit-shas"}:
        problems.append("scope.mode must be all-source-only or explicit-shas")

    review = config.get("review")
    if (
        not isinstance(review, dict)
        or review.get("mode") != "automatic-independent-agent"
    ):
        problems.append("review.mode must be automatic-independent-agent")
    elif review.get("require_endorsement") is not True:
        problems.append("review.require_endorsement must be true")

    approvals = config.get("approvals")
    if not isinstance(approvals, dict):
        problems.append("approvals must be an object")
    else:
        owner = approvals.get("scope_approved_by")
        if not isinstance(owner, str) or not owner.strip() or owner == "<human-owner>":
            problems.append("approvals.scope_approved_by must name the approving human")
        if approvals.get("remote_publication_approved") is not True:
            problems.append("approvals.remote_publication_approved must be true")
        if approvals.get("merge_authorized") is not False:
            problems.append("approvals.merge_authorized must be false")

    intent_file = scope.get("intent_file") if isinstance(scope, dict) else None
    if not isinstance(intent_file, str) or not intent_file.strip():
        problems.append("scope.intent_file must be non-empty")
    elif not (repo / intent_file).is_file():
        problems.append(f"scope.intent_file does not exist: {intent_file}")

    worktree = config.get("worktree")
    if not isinstance(worktree, dict):
        problems.append("worktree must be an object")
    else:
        if (
            not isinstance(worktree.get("root"), str)
            or not Path(worktree["root"]).is_absolute()
        ):
            problems.append("worktree.root must be an absolute path")
        for field in ("lease_hours", "retain_days", "active_lease_wait_minutes"):
            if type(worktree.get(field)) is not int or worktree[field] <= 0:
                problems.append(f"worktree.{field} must be a positive integer")
        if worktree.get("active_lease_policy") not in {"wait", "block"}:
            problems.append("worktree.active_lease_policy must be wait or block")

    if isinstance(scope, dict):
        if scope.get("new_source_commits") not in {"audit-and-continue", "block"}:
            problems.append(
                "scope.new_source_commits must be audit-and-continue or block"
            )
        if scope.get("unknown_source_author") not in {"block", "allow"}:
            problems.append("scope.unknown_source_author must be block or allow")
        approved_authors = scope.get("approved_source_authors")
        if not isinstance(approved_authors, list) or any(
            not isinstance(author, str) or not author.strip()
            for author in approved_authors
        ):
            problems.append("scope.approved_source_authors must be a string list")

    identity = config.get("identity")
    if not isinstance(identity, dict):
        problems.append("identity must be an object")
    else:
        developer = identity.get("porting_developer")
        if (
            not isinstance(developer, str)
            or re.fullmatch(r".+ <[^<>@\s]+@[^<>\s]+>", developer) is None
        ):
            problems.append("identity.porting_developer must be 'Name <email>'")

    environment = config.get("environment")
    if not isinstance(environment, dict):
        problems.append("environment must be an object")
    else:
        if (
            not isinstance(environment.get("fingerprint"), str)
            or re.fullmatch(r"[0-9a-f]{64}", environment["fingerprint"]) is None
        ):
            problems.append("environment.fingerprint must be a SHA256")
        if environment.get("allowed_kernel_provider") not in {
            "source",
            "precompiled",
            "either",
        }:
            problems.append(
                "environment.allowed_kernel_provider must be source, precompiled or either"
            )
        if environment.get("temporary_requirement_edits") not in {
            "allow-if-restored",
            "block",
        }:
            problems.append(
                "environment.temporary_requirement_edits must be allow-if-restored or block"
            )
        if environment.get("environment_mismatch") not in {
            "block",
            "revalidate-and-continue",
        }:
            problems.append(
                "environment.environment_mismatch must be block or revalidate-and-continue"
            )

    if isinstance(review, dict):
        if type(review.get("max_rounds")) is not int or review["max_rounds"] <= 0:
            problems.append("review.max_rounds must be a positive integer")
        if review.get("on_changes_required") not in {
            "fix-and-review-again",
            "block",
        }:
            problems.append(
                "review.on_changes_required must be fix-and-review-again or block"
            )
        if review.get("on_exhausted") not in {
            "block",
            "publish-draft-pending",
        }:
            problems.append(
                "review.on_exhausted must be block or publish-draft-pending"
            )
        if review.get("new_material_risk") not in {
            "block",
            "publish-draft-pending",
        }:
            problems.append(
                "review.new_material_risk must be block or publish-draft-pending"
            )

    validation = config.get("validation")
    if not isinstance(validation, dict):
        problems.append("validation must be an object")
    else:
        categories = validation.get("known_pending_categories")
        if not isinstance(categories, list) or any(
            not isinstance(item, str) or not item.strip() for item in categories
        ):
            problems.append("validation.known_pending_categories must be a string list")
        if validation.get("unknown_failure") != "block":
            problems.append("validation.unknown_failure must be block")
        if not isinstance(validation.get("publish_pending_verification"), bool):
            problems.append("validation.publish_pending_verification must be boolean")

    publication = config.get("publication")
    if not isinstance(publication, dict):
        problems.append("publication must be an object")
    else:
        if publication.get("mode") != "auto" or publication.get("provider") != "auto":
            problems.append("publication mode/provider must be auto")
        if not isinstance(publication.get("github_draft"), bool):
            problems.append("publication.github_draft must be boolean")
        if not isinstance(publication.get("publish_with_pending_review"), bool):
            problems.append("publication.publish_with_pending_review must be boolean")
        if not isinstance(publication.get("reuse_open_pr"), bool):
            problems.append("publication.reuse_open_pr must be boolean")
        if publication.get("overwrite_manual_pr_body") is not False:
            problems.append("publication.overwrite_manual_pr_body must be false")

    if isinstance(approvals, dict) and not isinstance(
        approvals.get("draft_publication_with_pending_approved"), bool
    ):
        problems.append(
            "approvals.draft_publication_with_pending_approved must be boolean"
        )

    retry = config.get("retry")
    if not isinstance(retry, dict):
        problems.append("retry must be an object")
    else:
        if retry.get("target_moved") not in {"reaudit-once", "block"}:
            problems.append("retry.target_moved must be reaudit-once or block")
        if retry.get("push_lease_conflict") not in {
            "refetch-and-reaudit",
            "block",
        }:
            problems.append(
                "retry.push_lease_conflict must be refetch-and-reaudit or block"
            )
        for field in ("max_retries", "max_runtime_minutes"):
            if type(retry.get(field)) is not int or retry[field] < 0:
                problems.append(f"retry.{field} must be a non-negative integer")

    if problems:
        raise ValueError("; ".join(problems))
    return config


def detect(repo: Path) -> dict[str, Any]:
    repo = repo.resolve()
    path = config_path(repo)
    candidates = branch_candidates(repo)
    recommended_authors: list[str] = []
    if candidates["latest_mx_dev"] and candidates["master_available"]:
        try:
            recommended_authors = source_authors(
                repo, candidates["latest_mx_dev"], "master"
            )
        except (ValueError, RuntimeError):
            recommended_authors = []
    candidates["recommended_source_authors"] = recommended_authors
    if path.exists():
        config = json.loads(path.read_text(encoding="utf-8"))
        validate_config(repo, config)
        return {
            "status": "configured",
            "config_path": str(path),
            "scope_path": str(scope_path(repo)),
            "config": config,
            "detected": candidates,
            "questions": [],
        }

    questions = [
        {
            "id": "source_ref",
            "question": "Which source branch/range should be ported each day?",
            "recommended": candidates["latest_mx_dev"],
            "candidates": candidates["mx_dev_branches"],
        },
        {
            "id": "target_ref",
            "question": "Which target branch should receive the port?",
            "recommended": "master" if candidates["master_available"] else None,
            "candidates": ["master"],
        },
        {
            "id": "scope_mode",
            "question": "Port every source-only commit or an explicit SHA subset?",
            "recommended": "all-source-only",
            "candidates": ["all-source-only", "explicit-shas"],
        },
        {
            "id": "remote_publication_approved",
            "question": "Approve automatic Gerrit/GitHub publication for this bounded scope?",
            "recommended": None,
            "candidates": [True, False],
        },
        {
            "id": "scope_approved_by",
            "question": "Who is the accountable human owner approving this daily scope?",
            "recommended": None,
            "candidates": [],
        },
        {
            "id": "porting_developer",
            "question": "Which Git identity should author and commit master ports?",
            "recommended": None,
            "candidates": [],
        },
        {
            "id": "source_author_policy",
            "question": "How should source-author permission be handled for future commits?",
            "recommended": (
                "current-branch-authors" if recommended_authors else "block-unknown"
            ),
            "candidates": [
                "current-branch-authors",
                "block-unknown",
                "allow-all",
            ],
            "detected_authors": recommended_authors,
        },
        {
            "id": "draft_publication_with_pending_approved",
            "question": "Allow draft Gerrit/GitHub publication when only pre-approved pending validation remains?",
            "recommended": True,
            "candidates": [True, False],
        },
    ]
    return {
        "status": "missing",
        "config_path": str(path),
        "scope_path": str(scope_path(repo)),
        "detected": candidates,
        "questions": questions,
        "safe_defaults": {
            "publication": {"mode": "auto", "provider": "auto", "github_draft": True},
            "review": {
                "mode": "automatic-independent-agent",
                "require_endorsement": True,
                "max_rounds": 2,
                "on_changes_required": "fix-and-review-again",
                "on_exhausted": "block",
                "new_material_risk": "block",
            },
            "validation": {
                "known_pending_categories": [
                    "performance-unmeasured",
                    "hardware-unavailable",
                    "model-unavailable",
                ],
                "unknown_failure": "block",
                "publish_pending_verification": True,
            },
            "worktree": {
                "root": str(repo.parent / ".metax-worktrees"),
                "lease_hours": 24,
                "retain_days": 14,
                "active_lease_policy": "wait",
                "active_lease_wait_minutes": 30,
            },
            "retry": {
                "target_moved": "reaudit-once",
                "push_lease_conflict": "refetch-and-reaudit",
                "max_retries": 1,
                "max_runtime_minutes": 360,
            },
        },
    }


def create_from_answers(
    repo: Path, answers_path: Path, *, force: bool
) -> dict[str, Any]:
    repo = repo.resolve()
    path = config_path(repo)
    if path.exists() and not force:
        raise ValueError(f"daily config already exists: {path}")
    answers = json.loads(answers_path.read_text(encoding="utf-8"))
    scope_mode = answers.get("scope_mode")
    if scope_mode not in {"all-source-only", "explicit-shas"}:
        raise ValueError("scope_mode must be all-source-only or explicit-shas")
    source_shas = answers.get("source_shas", [])
    exclude_source_shas = answers.get("exclude_source_shas", [])
    if not isinstance(source_shas, list) or not isinstance(exclude_source_shas, list):
        raise ValueError("source_shas and exclude_source_shas must be lists")
    if scope_mode == "explicit-shas" and not source_shas:
        raise ValueError("explicit-shas mode requires at least one source SHA")
    for sha in [*source_shas, *exclude_source_shas]:
        if not isinstance(sha, str) or SHA_RE.fullmatch(sha) is None:
            raise ValueError(f"source/range contains an invalid full SHA: {sha!r}")

    source_author_policy = answers.get("source_author_policy", "block-unknown")
    if source_author_policy not in {
        "current-branch-authors",
        "block-unknown",
        "allow-all",
    }:
        raise ValueError(
            "source_author_policy must be current-branch-authors, block-unknown or allow-all"
        )
    approved_source_authors = answers.get("approved_source_authors", [])
    if source_author_policy == "current-branch-authors":
        approved_source_authors = source_authors(
            repo, answers["source_ref"], answers["target_ref"]
        )
    elif source_author_policy == "allow-all":
        approved_source_authors = []

    scope_file = scope_path(repo)
    scope = {
        "schema_version": 1,
        "mode": scope_mode,
        "source_shas": source_shas,
        "exclude_source_shas": exclude_source_shas,
        "intent_ids": answers.get("intent_ids", []),
        "notes": answers.get(
            "notes", "Created from daily configuration questionnaire."
        ),
    }
    relative_scope = str(scope_file.relative_to(repo))
    worktree_root = answers.get("worktree_root", str(repo.parent / ".metax-worktrees"))
    environment_fingerprint = (
        answers.get("environment_fingerprint") or compute_fingerprint()["fingerprint"]
    )
    config = {
        "schema_version": 1,
        "source_ref": answers["source_ref"],
        "target_ref": answers["target_ref"],
        "scope": {
            "mode": scope_mode,
            "intent_file": relative_scope,
            "new_source_commits": answers.get(
                "new_source_commits", "audit-and-continue"
            ),
            "unknown_source_author": (
                "allow" if source_author_policy == "allow-all" else "block"
            ),
            "source_author_policy": source_author_policy,
            "approved_source_authors": approved_source_authors,
        },
        "identity": {
            "porting_developer": answers["porting_developer"],
        },
        "environment": {
            "fingerprint": environment_fingerprint,
            "allowed_kernel_provider": answers.get("allowed_kernel_provider", "source"),
            "temporary_requirement_edits": answers.get(
                "temporary_requirement_edits", "allow-if-restored"
            ),
            "environment_mismatch": answers.get("environment_mismatch", "block"),
        },
        "publication": {
            "mode": "auto",
            "provider": "auto",
            "github_draft": bool(answers.get("github_draft", True)),
            "publish_with_pending_review": bool(
                answers.get("publish_with_pending_review", True)
            ),
            "reuse_open_pr": bool(answers.get("reuse_open_pr", True)),
            "overwrite_manual_pr_body": False,
        },
        "review": {
            "mode": "automatic-independent-agent",
            "require_endorsement": True,
            "max_rounds": int(answers.get("review_max_rounds", 2)),
            "on_changes_required": answers.get(
                "on_changes_required", "fix-and-review-again"
            ),
            "on_exhausted": answers.get("on_exhausted", "block"),
            "new_material_risk": answers.get("new_material_risk", "block"),
        },
        "validation": {
            "known_pending_categories": answers.get(
                "known_pending_categories",
                [
                    "performance-unmeasured",
                    "hardware-unavailable",
                    "model-unavailable",
                ],
            ),
            "unknown_failure": "block",
            "publish_pending_verification": bool(
                answers.get("publish_pending_verification", True)
            ),
        },
        "worktree": {
            "root": worktree_root,
            "lease_hours": int(answers.get("lease_hours", 24)),
            "retain_days": int(answers.get("retain_days", 14)),
            "active_lease_policy": answers.get("active_lease_policy", "wait"),
            "active_lease_wait_minutes": int(
                answers.get("active_lease_wait_minutes", 30)
            ),
        },
        "retry": {
            "target_moved": answers.get("target_moved", "reaudit-once"),
            "push_lease_conflict": answers.get(
                "push_lease_conflict", "refetch-and-reaudit"
            ),
            "max_retries": int(answers.get("max_retries", 1)),
            "max_runtime_minutes": int(answers.get("max_runtime_minutes", 360)),
        },
        "approvals": {
            "scope_approved_by": answers["scope_approved_by"],
            "remote_publication_approved": answers["remote_publication_approved"],
            "draft_publication_with_pending_approved": bool(
                answers.get("draft_publication_with_pending_approved", True)
            ),
            "merge_authorized": False,
        },
    }
    atomic_json(scope_file, scope)
    atomic_json(path, config)
    validate_config(repo, config)
    return {
        "status": "created",
        "config_path": str(path),
        "scope_path": str(scope_file),
        "config": config,
        "scope": scope,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    detect_parser = commands.add_parser("detect")
    detect_parser.add_argument("--repo", type=Path, required=True)
    create_parser = commands.add_parser("create")
    create_parser.add_argument("--repo", type=Path, required=True)
    create_parser.add_argument("--answers", type=Path, required=True)
    create_parser.add_argument("--force", action="store_true")
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        if args.action == "detect":
            result = detect(args.repo)
        elif args.action == "create":
            result = create_from_answers(args.repo, args.answers, force=args.force)
        else:
            repo = args.repo.resolve()
            config = json.loads(config_path(repo).read_text(encoding="utf-8"))
            result = {
                "status": "valid",
                "config": validate_config(repo, config),
            }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError, KeyError) as error:
        print(f"daily config blocked: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
