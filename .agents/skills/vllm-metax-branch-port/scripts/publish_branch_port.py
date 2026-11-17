#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Inspect and publish approved branch-port results without exposing repository URLs."""

from __future__ import annotations

import argparse
import ast
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any
from collections.abc import Sequence

import regex as re

MX_BRANCH_RE = re.compile(r"^mx/v(?P<version>\d+\.\d+\.\d+)-dev$")
URL_RE = re.compile(
    r"(?:https?|ssh|git)://\S+|git@[^:\s]+:\S+|(?:www\.)?github\.com[/:]\S+",
    re.IGNORECASE,
)
CHANGE_ID_RE = re.compile(r"^Change-Id:\s*(\S+)\s*$", re.MULTILINE)
SOURCE_RE = re.compile(r"^Source:\s*\S+\s*$", re.MULTILINE)
PR_NUMBER_RE = re.compile(r"/pull/(\d+)")


def now() -> str:
    return datetime.now(UTC).isoformat()


def redact(text: str) -> str:
    return URL_RE.sub("<redacted-remote>", text)


def run(
    repo: Path,
    *args: str,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        env=env,
    )
    if check and result.returncode:
        raise RuntimeError(
            redact(
                f"git {' '.join(args)} failed:\n{result.stdout}{result.stderr}".strip()
            )
        )
    return result


def run_gh(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    gh = shutil.which("gh")
    if gh is None:
        raise RuntimeError(
            "GitHub publication requires an installed and authenticated gh"
        )
    result = subprocess.run(
        [gh, *args],
        text=True,
        capture_output=True,
    )
    if check and result.returncode:
        raise RuntimeError(
            redact(
                f"gh {' '.join(args)} failed:\n{result.stdout}{result.stderr}".strip()
            )
        )
    return result


def git_output(repo: Path, *args: str) -> str:
    return run(repo, *args).stdout.strip()


def remote_urls(repo: Path, remote: str) -> list[str]:
    result = run(repo, "remote", "get-url", "--all", remote)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def remote_kind(repo: Path, remote: str) -> str:
    urls = remote_urls(repo, remote)
    if any(re.search(r"github\.com[/:]", url, re.IGNORECASE) for url in urls):
        return "github"
    return "other"


def parse_mx_branch(branch: str) -> tuple[int, int, int] | None:
    match = MX_BRANCH_RE.fullmatch(branch)
    if match is None:
        return None
    return tuple(int(part) for part in match.group("version").split("."))


def remote_branches(repo: Path) -> list[tuple[str, str]]:
    output = git_output(
        repo,
        "for-each-ref",
        "--format=%(refname)",
        "refs/remotes",
    )
    branches = []
    for ref in output.splitlines():
        prefix = "refs/remotes/"
        if not ref.startswith(prefix):
            continue
        remote, separator, branch = ref[len(prefix) :].partition("/")
        if separator and branch:
            branches.append((remote, branch))
    return branches


def find_remote_for_branch(
    repo: Path,
    branch: str,
    *,
    required_kind: str | None = None,
) -> str | None:
    candidates = []
    for remote, remote_branch in remote_branches(repo):
        if remote_branch != branch:
            continue
        kind = remote_kind(repo, remote)
        matches = (
            required_kind is None
            or kind == required_kind
            or (required_kind == "gerrit" and kind != "github")
        )
        if matches:
            candidates.append((remote, kind))
    if len(candidates) == 1:
        return candidates[0][0]
    if required_kind == "gerrit":
        non_github = [remote for remote, kind in candidates if kind != "github"]
        if len(non_github) == 1:
            return non_github[0]
    return None


def branch_sha(repo: Path, remote: str, branch: str) -> str:
    return git_output(
        repo,
        "rev-parse",
        "--verify",
        f"refs/remotes/{remote}/{branch}^{{commit}}",
    )


def fixed_public_version(repo: Path, sha: str) -> str:
    try:
        source = run(repo, "show", f"{sha}:setup.py").stdout
    except RuntimeError as error:
        raise RuntimeError(
            f"Cannot read setup.py for the selected ref: {error}"
        ) from error
    tree = ast.parse(source)
    returns = [
        node
        for function in tree.body
        if isinstance(function, ast.FunctionDef)
        and function.name == "fixed_version_scheme"
        for node in ast.walk(function)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ]
    if len(returns) != 1:
        raise RuntimeError("setup.py must have one fixed_version_scheme string return")
    version = returns[0].value.value
    if re.fullmatch(r"\d+\.\d+\.\d+", version) is None:
        raise RuntimeError(f"Invalid public version in setup.py: {version!r}")
    return version


def version_tuple(version: str) -> tuple[int, int, int]:
    return tuple(int(part) for part in version.split("."))


def latest_mx_version(repo: Path) -> tuple[str, tuple[int, int, int]] | None:
    versions: dict[str, tuple[int, int, int]] = {}
    for _, branch in remote_branches(repo):
        version = parse_mx_branch(branch)
        if version is not None:
            versions[branch[len("mx/v") : -len("-dev")]] = version
    if not versions:
        return None
    text, value = max(versions.items(), key=lambda item: item[1])
    return text, value


@dataclass
class Selection:
    branch: str
    provider: str
    remote: str | None
    sha: str | None
    public_version: str | None


def inspect_branch(repo: Path, branch: str) -> Selection:
    parsed_mx = parse_mx_branch(branch)
    if branch.startswith("mx/") and parsed_mx is None:
        return Selection(branch, "blocked", None, None, None)
    if parsed_mx is not None:
        remote = find_remote_for_branch(repo, branch, required_kind="gerrit")
        sha = branch_sha(repo, remote, branch) if remote else None
        public_version = ".".join(str(part) for part in parsed_mx)
        if remote and sha:
            setup_version = fixed_public_version(repo, sha)
            if setup_version != public_version:
                return Selection(branch, "blocked", remote, sha, setup_version)
        return Selection(branch, "gerrit", remote, sha, public_version)
    if branch == "master":
        remote = find_remote_for_branch(repo, branch, required_kind="github")
        if remote:
            sha = branch_sha(repo, remote, branch)
            provider = "github"
        else:
            local_ref = "refs/heads/master"
            result = run(
                repo, "rev-parse", "--verify", f"{local_ref}^{{commit}}", check=False
            )
            sha = result.stdout.strip() if result.returncode == 0 else None
            provider = "local" if sha else "blocked"
        version = fixed_public_version(repo, sha) if sha else None
        return Selection(branch, provider, remote, sha, version)
    return Selection(branch, "local", None, None, None)


def inspect_publication(repo: Path, source_ref: str, target_ref: str) -> dict[str, Any]:
    source = inspect_branch(repo, source_ref)
    target = inspect_branch(repo, target_ref)
    master = inspect_branch(repo, "master")
    latest = latest_mx_version(repo)
    issues = []
    if source.provider == "blocked":
        issues.append(f"Invalid source branch contract: {source_ref}")
    if source.provider == "gerrit" and source.remote is None:
        issues.append("No unique Gerrit remote owns the source mx branch")
    if target.provider == "blocked":
        issues.append(f"Invalid target branch contract: {target_ref}")
    if target.provider == "gerrit" and target.remote is None:
        issues.append("No unique Gerrit remote owns the target mx branch")
    if target.provider == "github" and target.remote is None:
        issues.append("No GitHub remote owns master")
    if master.provider == "blocked":
        issues.append("No local or GitHub master baseline is available")
    if latest is None:
        issues.append("No mx/vX.Y.Z-dev remote branch is available")
    master_gate = None
    if master.public_version and latest:
        master_gate = version_tuple(master.public_version) >= latest[1]
        if not master_gate:
            issues.append(
                f"master {master.public_version} is below latest mx dev {latest[0]}"
            )
    if target.provider == "gerrit" and target.remote:
        review_ref = f"refs/for/{target.branch}"
        publish_plan = {
            "provider": "gerrit",
            "remote": target.remote,
            "review_ref": review_ref,
        }
    elif target.provider == "github" and target.remote:
        publish_plan = {
            "provider": "github",
            "remote": target.remote,
            "base": "master",
            "head_branch": "<dedicated-pr-branch>",
        }
    else:
        publish_plan = {
            "provider": "local",
            "remote": None,
            "review_ref": None,
            "pr_base": None,
        }
    return {
        "schema_version": 1,
        "checked_at": now(),
        "source": source.__dict__,
        "target": target.__dict__,
        "final_target": "master",
        "master": master.__dict__,
        "latest_mx_dev": (
            {"version": latest[0], "tuple": latest[1]} if latest is not None else None
        ),
        "master_version_gate": master_gate,
        "local_only_reason": (
            "master-is-not-a-github-remote"
            if master.provider == "local" and target.provider == "local"
            else None
        ),
        "publish_plan": publish_plan,
        "master_commit_policy": {
            "author": "approved-porting-developer",
            "committer": "approved-porting-developer",
            "source_trailer": "absent",
        },
        "issues": issues,
        "status": "ok" if not issues else "blocked",
    }


def remote_mx_branches(repo: Path, remote: str) -> dict[str, str]:
    result = run(
        repo,
        "ls-remote",
        "--heads",
        remote,
        "refs/heads/mx/v*-dev",
    )
    branches: dict[str, str] = {}
    for line in result.stdout.splitlines():
        sha, separator, ref = line.partition("\t")
        prefix = "refs/heads/"
        if not separator or not ref.startswith(prefix):
            continue
        branch = ref[len(prefix) :]
        if parse_mx_branch(branch) is not None:
            branches[branch] = sha
    return branches


def tracking_mx_branches(repo: Path, remote: str) -> dict[str, str]:
    branches = {}
    for remote_name, branch in remote_branches(repo):
        if remote_name != remote or parse_mx_branch(branch) is None:
            continue
        branches[branch] = branch_sha(repo, remote_name, branch)
    return branches


def sorted_mx_chain(branches: dict[str, str]) -> list[str]:
    return sorted(
        branches,
        key=lambda branch: parse_mx_branch(branch) or (0, 0, 0),
    )


def requested_chain(
    chain: list[str],
    source_ref: str,
    final_target: str,
    chain_arg: str | None,
) -> list[str]:
    if chain_arg:
        chain = [
            item.strip()
            for item in re.split(r"\s*(?:->|,)\s*", chain_arg)
            if item.strip()
        ]
    try:
        start = chain.index(source_ref)
        end = chain.index(final_target)
    except ValueError as error:
        raise RuntimeError(
            f"source {source_ref!r} and final target {final_target!r} must both be "
            "present in the configured or explicit chain"
        ) from error
    if end <= start:
        raise RuntimeError("final target must be later than the source branch")
    return chain[start : end + 1]


def plan_chain(
    repo: Path,
    source_ref: str,
    final_target: str,
    chain_arg: str | None,
    *,
    use_tracking_refs: bool = False,
) -> dict[str, Any]:
    source_remote = find_remote_for_branch(repo, source_ref, required_kind="gerrit")
    if source_remote is None:
        raise RuntimeError(f"No unique Gerrit remote owns source {source_ref}")
    discovered = (
        tracking_mx_branches(repo, source_remote)
        if use_tracking_refs
        else remote_mx_branches(repo, source_remote)
    )
    if source_ref not in discovered:
        raise RuntimeError(
            f"Source branch {source_ref} does not exist on its Gerrit remote"
        )
    discovered_chain = [*sorted_mx_chain(discovered), "master"]
    full_chain = (
        [
            item.strip()
            for item in re.split(r"\s*(?:->|,)\s*", chain_arg)
            if item.strip()
        ]
        if chain_arg
        else discovered_chain
    )
    chain = requested_chain(full_chain, source_ref, final_target, chain_arg)
    remote_mx = set(discovered)
    source_version = parse_mx_branch(source_ref)
    if final_target == "master":
        master = inspect_branch(repo, "master")
        target_version = (
            version_tuple(master.public_version) if master.public_version else None
        )
    else:
        target_version = parse_mx_branch(final_target)
    skipped = []
    if source_version is not None and target_version is not None:
        skipped = sorted(
            branch
            for branch in remote_mx.difference(full_chain)
            if (branch_version := parse_mx_branch(branch)) is not None
            and source_version < branch_version <= target_version
        )
    edges = []
    for source, target in zip(chain, chain[1:], strict=False):
        edge = inspect_publication(repo, source, target)
        edges.append(
            {
                "source_ref": source,
                "target_ref": target,
                "source_sha": edge["source"]["sha"],
                "target_sha": edge["target"]["sha"],
                "source_remote": edge["source"]["remote"],
                "target_remote": edge["target"]["remote"],
                "provider": edge["target"]["provider"],
                "target_public_version": edge["target"]["public_version"],
                "publish_plan": edge["publish_plan"],
                "ready": edge["status"] == "ok",
                "issues": edge["issues"],
            }
        )
    issues = [
        f"{edge['source_ref']} -> {edge['target_ref']}: {issue}"
        for edge in edges
        for issue in edge["issues"]
    ]
    return {
        "schema_version": 1,
        "checked_at": now(),
        "source_ref": source_ref,
        "final_target": final_target,
        "configured_chain": full_chain,
        "discovered_mx_branches": discovered,
        "branch_inventory_source": (
            "remote_tracking_refs" if use_tracking_refs else "live_remote"
        ),
        "discovered_chain": discovered_chain,
        "resolved_chain": chain,
        "skipped_mx_branches": skipped,
        "edges": edges,
        "chain_status": "ready" if not issues else "blocked",
        "issues": issues,
        "confirmation_required": {
            "branch_chain": True,
            "source_and_target_remotes": True,
            "remote_publication": True,
        },
    }


def commit_range(repo: Path, base: str, tip: str = "HEAD") -> list[str]:
    return git_output(repo, "rev-list", "--reverse", f"{base}..{tip}").splitlines()


def commit_message(repo: Path, sha: str) -> str:
    return git_output(repo, "show", "-s", "--format=%B", sha)


def commit_identity(repo: Path, sha: str) -> tuple[str, str]:
    output = git_output(
        repo,
        "show",
        "-s",
        "--format=%an%x00%ae%x00%cn%x00%ce",
        sha,
    )
    author_name, author_email, committer_name, committer_email = output.split("\0")
    return f"{author_name} <{author_email}>", f"{committer_name} <{committer_email}>"


def validate_gerrit_commits(repo: Path, base: str) -> None:
    merges = git_output(repo, "rev-list", "--merges", f"{base}..HEAD").splitlines()
    if merges:
        raise RuntimeError("Gerrit publication does not accept local merge commits")
    change_ids = []
    for sha in commit_range(repo, base):
        ids = CHANGE_ID_RE.findall(commit_message(repo, sha))
        if len(ids) != 1:
            raise RuntimeError(f"Commit {sha} must contain exactly one Change-Id")
        change_ids.append(ids[0])
    if len(change_ids) != len(set(change_ids)):
        raise RuntimeError("Gerrit publication requires unique Change-Ids")


def validate_master_commits(repo: Path, base: str, identity: str) -> None:
    for sha in commit_range(repo, base):
        author, committer = commit_identity(repo, sha)
        if author != identity or committer != identity:
            raise RuntimeError(f"Master commit {sha} identity differs from {identity}")
        if SOURCE_RE.search(commit_message(repo, sha)):
            raise RuntimeError(f"Master commit {sha} must not contain Source:")


def remote_branch_sha(repo: Path, remote: str, branch: str) -> str | None:
    result = run(
        repo,
        "ls-remote",
        "--heads",
        remote,
        f"refs/heads/{branch}",
    )
    if not result.stdout.strip():
        return None
    return result.stdout.split()[0]


def github_slug(repo: Path, remote: str) -> tuple[str, str]:
    url = remote_urls(repo, remote)[0]
    result = run_gh("repo", "view", url, "--json", "nameWithOwner")
    slug = json.loads(result.stdout)["nameWithOwner"]
    owner_match = re.search(r"github\.com[/:]([^/]+)/([^/]+?)(?:\.git)?$", url)
    if owner_match is None:
        raise RuntimeError("Cannot resolve the GitHub remote owner")
    return slug, owner_match.group(1)


def write_receipt(run_dir: Path, receipt: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "PUBLISH.json").write_text(
        json.dumps(receipt, indent=2, ensure_ascii=False) + "\n"
    )


def publish(args: argparse.Namespace) -> int:
    repo = args.repo.resolve()
    run_dir = args.run_dir.resolve()
    report = inspect_publication(repo, args.source_ref, args.target_ref)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "started_at": now(),
        "target_ref": args.target_ref,
        "inspection": report,
        "status": "blocked",
    }
    if report["status"] != "ok":
        write_receipt(run_dir, receipt)
        print(json.dumps(receipt, indent=2, ensure_ascii=False))
        return 1
    if git_output(repo, "status", "--porcelain"):
        raise RuntimeError("Publication requires a clean worktree")
    base = args.base_sha or git_output(repo, "rev-parse", f"{args.base_ref}^{{commit}}")
    provider = report["target"]["provider"]
    remote = report["target"]["remote"]
    if provider == "local":
        receipt.update(
            status="local-only",
            local_only_reason=report["local_only_reason"],
            output="No remote mutation requested or available.",
        )
    elif provider == "gerrit":
        validate_gerrit_commits(repo, base)
        review_ref = f"refs/for/{args.target_ref}"
        command = [
            "git",
            "-C",
            str(repo),
            "push",
            "--porcelain",
            remote,
            f"HEAD:{review_ref}",
        ]
        result = subprocess.run(command, text=True, capture_output=True)
        receipt.update(
            status="pushed" if result.returncode == 0 else "blocked",
            logical_remote=remote,
            review_ref=review_ref,
            output=redact(result.stdout + result.stderr),
        )
    elif provider == "github":
        if not args.pr_branch or not args.title:
            raise RuntimeError("GitHub publication requires --pr-branch and --title")
        if args.pr_branch == "master":
            raise RuntimeError("GitHub PR head branch must not be master")
        if not args.porting_identity:
            raise RuntimeError("GitHub publication requires --porting-identity")
        validate_master_commits(repo, base, args.porting_identity)
        run_gh("auth", "status")
        slug, owner = github_slug(repo, remote)
        remote_sha = remote_branch_sha(repo, remote, args.pr_branch)
        head_sha = git_output(repo, "rev-parse", "HEAD")
        if remote_sha and remote_sha != head_sha:
            ancestor = run(
                repo,
                "merge-base",
                "--is-ancestor",
                remote_sha,
                head_sha,
                check=False,
            )
            if ancestor.returncode:
                raise RuntimeError(
                    "Existing GitHub PR branch is not a fast-forward of local HEAD"
                )
        result = run(
            repo,
            "push",
            "--porcelain",
            remote,
            f"HEAD:refs/heads/{args.pr_branch}",
        )
        head_ref = (
            args.pr_branch
            if owner.lower() == slug.split("/", 1)[0].lower()
            else f"{owner}:{args.pr_branch}"
        )
        existing = run_gh(
            "pr",
            "list",
            "--repo",
            slug,
            "--head",
            head_ref,
            "--base",
            "master",
            "--state",
            "open",
            "--json",
            "number",
        )
        prs = json.loads(existing.stdout)
        if prs:
            number = prs[0]["number"]
            status = "pr-open"
        else:
            body_file = args.body_file
            if body_file is None:
                body_file = run_dir / "VALIDATION.md"
            created = run_gh(
                "pr",
                "create",
                "--repo",
                slug,
                "--base",
                "master",
                "--head",
                head_ref,
                "--title",
                args.title,
                "--body-file",
                str(body_file),
            )
            match = PR_NUMBER_RE.search(created.stdout)
            if match is None:
                raise RuntimeError("GitHub PR creation did not return a PR number")
            number = int(match.group(1))
            status = "pr-open"
        receipt.update(
            status=status,
            logical_remote=remote,
            base="master",
            head_branch=args.pr_branch,
            pr_number=number,
            push_output=redact(result.stdout + result.stderr),
        )
    else:
        raise RuntimeError("Remote publication is unavailable for this target")
    receipt["finished_at"] = now()
    write_receipt(run_dir, receipt)
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    inspect_parser = commands.add_parser("inspect")
    inspect_parser.add_argument("--repo", type=Path, required=True)
    inspect_parser.add_argument("--source-ref", required=True)
    inspect_parser.add_argument("--target-ref", required=True)
    chain_parser = commands.add_parser("plan-chain")
    chain_parser.add_argument("--repo", type=Path, required=True)
    chain_parser.add_argument("--source-ref", required=True)
    chain_parser.add_argument("--final-target", default="master")
    chain_parser.add_argument("--chain")
    chain_parser.add_argument("--use-tracking-refs", action="store_true")
    publish_parser = commands.add_parser("publish")
    publish_parser.add_argument("--repo", type=Path, required=True)
    publish_parser.add_argument("--run-dir", type=Path, required=True)
    publish_parser.add_argument("--source-ref", required=True)
    publish_parser.add_argument("--target-ref", required=True)
    publish_parser.add_argument("--base-ref", default="HEAD^")
    publish_parser.add_argument("--base-sha")
    publish_parser.add_argument("--pr-branch")
    publish_parser.add_argument("--title")
    publish_parser.add_argument("--body-file", type=Path)
    publish_parser.add_argument("--porting-identity")
    args = parser.parse_args(argv)
    try:
        if args.action == "inspect":
            print(
                json.dumps(
                    inspect_publication(
                        args.repo.resolve(), args.source_ref, args.target_ref
                    ),
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0
        if args.action == "plan-chain":
            print(
                json.dumps(
                    plan_chain(
                        args.repo.resolve(),
                        args.source_ref,
                        args.final_target,
                        args.chain,
                        use_tracking_refs=args.use_tracking_refs,
                    ),
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0
        return publish(args)
    except (RuntimeError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"publication blocked: {redact(str(error))}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
