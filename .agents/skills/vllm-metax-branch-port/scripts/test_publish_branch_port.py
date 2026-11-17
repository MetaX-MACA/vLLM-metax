# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import subprocess
from pathlib import Path

import publish_branch_port as publish


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
    ).strip()


def commit_setup(repo: Path, version: str, message: str) -> str:
    (repo / "setup.py").write_text(
        f'def fixed_version_scheme(version):\n    return "{version}"\n'
    )
    git(repo, "add", "setup.py")
    git(repo, "commit", "-m", message)
    return git(repo, "rev-parse", "HEAD")


def make_repo(
    tmp_path: Path,
    master_version: str,
    *,
    github_remote: bool = True,
) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-b", "master")
    git(repo, "config", "user.name", "Port Developer")
    git(repo, "config", "user.email", "port@example.com")
    master_sha = commit_setup(repo, master_version, "master")
    git(repo, "checkout", "-b", "mx/v0.34.0-dev")
    branch_sha = commit_setup(repo, "0.34.0", "dev")
    git(repo, "checkout", "master")
    git(repo, "remote", "add", "review", "https://gerrit.example/a/project")
    git(
        repo,
        "update-ref",
        "refs/remotes/review/mx/v0.34.0-dev",
        branch_sha,
    )
    if github_remote:
        git(repo, "remote", "add", "github", "https://github.com/org/repo.git")
        git(repo, "update-ref", "refs/remotes/github/master", master_sha)
    return repo


def test_mx_branch_requires_strict_version_contract():
    assert publish.parse_mx_branch("mx/v0.34.0-dev") == (0, 34, 0)
    assert publish.parse_mx_branch("mx/v0.34-dev") is None
    assert publish.parse_mx_branch("mx/feature") is None
    assert publish.parse_mx_branch("master") is None


def test_dynamic_chain_sorts_versions_and_skips_missing_branches():
    discovered = publish.sorted_mx_chain(
        {
            "mx/v0.32.0-dev": "c" * 40,
            "mx/v0.30.0-dev": "a" * 40,
        }
    )
    chain = [*discovered, "master"]

    assert discovered == ["mx/v0.30.0-dev", "mx/v0.32.0-dev"]
    assert publish.requested_chain(
        chain,
        "mx/v0.30.0-dev",
        "master",
        None,
    ) == ["mx/v0.30.0-dev", "mx/v0.32.0-dev", "master"]
    assert publish.requested_chain(
        chain,
        "mx/v0.32.0-dev",
        "master",
        None,
    ) == [
        "mx/v0.32.0-dev",
        "master",
    ]


def test_explicit_chain_can_override_default_matrix():
    resolved = publish.requested_chain(
        ["mx/v0.34.0-dev", "master"],
        "mx/v0.34.0-dev",
        "master",
        "mx/v0.34.0-dev -> master",
    )

    assert resolved == ["mx/v0.34.0-dev", "master"]


def test_publication_plan_uses_gerrit_for_mx_and_github_for_master(tmp_path):
    repo = make_repo(tmp_path, "0.35.0")

    report = publish.inspect_publication(repo, "mx/v0.34.0-dev", "master")

    assert report["status"] == "ok"
    assert report["source"]["provider"] == "gerrit"
    assert report["target"]["provider"] == "github"
    assert report["publish_plan"] == {
        "provider": "github",
        "remote": "github",
        "base": "master",
        "head_branch": "<dedicated-pr-branch>",
    }
    assert report["master_version_gate"] is True


def test_plan_chain_shows_every_edge_before_confirmation(tmp_path, monkeypatch):
    repo = make_repo(tmp_path, "0.35.0")
    branch_sha = git(repo, "rev-parse", "refs/remotes/review/mx/v0.34.0-dev")
    monkeypatch.setattr(
        publish,
        "remote_mx_branches",
        lambda repo, remote: {"mx/v0.34.0-dev": branch_sha},
    )

    plan = publish.plan_chain(
        repo,
        "mx/v0.34.0-dev",
        "master",
        "mx/v0.34.0-dev -> master",
    )

    assert plan["chain_status"] == "ready"
    assert plan["discovered_chain"] == ["mx/v0.34.0-dev", "master"]
    assert plan["resolved_chain"] == ["mx/v0.34.0-dev", "master"]
    assert plan["edges"][0]["source_ref"] == "mx/v0.34.0-dev"
    assert plan["edges"][0]["target_ref"] == "master"
    assert plan["edges"][0]["publish_plan"]["provider"] == "github"
    assert plan["confirmation_required"] == {
        "branch_chain": True,
        "source_and_target_remotes": True,
        "remote_publication": True,
    }


def test_plan_chain_can_use_refreshed_tracking_refs(tmp_path):
    repo = make_repo(tmp_path, "0.35.0")

    plan = publish.plan_chain(
        repo,
        "mx/v0.34.0-dev",
        "master",
        None,
        use_tracking_refs=True,
    )

    assert plan["branch_inventory_source"] == "remote_tracking_refs"
    assert plan["resolved_chain"] == ["mx/v0.34.0-dev", "master"]


def test_master_version_must_cover_latest_mx_dev(tmp_path):
    repo = make_repo(tmp_path, "0.33.0")

    report = publish.inspect_publication(repo, "mx/v0.34.0-dev", "master")

    assert report["status"] == "blocked"
    assert report["master_version_gate"] is False


def test_master_without_github_remote_is_local_only(tmp_path):
    repo = make_repo(tmp_path, "0.35.0", github_remote=False)

    report = publish.inspect_publication(repo, "mx/v0.34.0-dev", "master")

    assert report["status"] == "ok"
    assert report["master"]["provider"] == "local"
    assert report["publish_plan"]["provider"] == "local"
    assert report["local_only_reason"] == "master-is-not-a-github-remote"


def test_master_commit_validation_rejects_source_and_other_author(tmp_path):
    repo = make_repo(tmp_path, "0.35.0")
    base = git(repo, "rev-parse", "HEAD")
    (repo / "change.txt").write_text("change\n")
    git(repo, "add", "change.txt")
    git(
        repo,
        "-c",
        "user.name=Port Developer",
        "-c",
        "user.email=port@example.com",
        "commit",
        "--author=Other <other@example.com>",
        "-m",
        "change",
        "-m",
        "Source: mx/v0.34.0-dev@deadbeef",
    )

    try:
        publish.validate_master_commits(
            repo,
            base,
            "Port Developer <port@example.com>",
        )
    except RuntimeError as error:
        assert "identity" in str(error) or "Source" in str(error)
    else:
        raise AssertionError("invalid master commit must be rejected")


def test_remote_urls_are_redacted():
    output = (
        "To https://github.com/org/repo.git, "
        "git@github.com:org/repo.git and github.com/org/repo"
    )
    redacted = publish.redact(output)

    assert "github.com/org" not in redacted
    assert "<redacted-remote>" in redacted
