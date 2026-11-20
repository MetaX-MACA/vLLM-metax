#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Stable, repository-external paths for branch-port agent state."""

from __future__ import annotations

import os

import regex as re
from pathlib import Path

DEFAULT_AGENT_HOME = Path("/workspace/.metax-agent-state")
SAFE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_component(value: str) -> str:
    value = SAFE_COMPONENT_RE.sub("-", value.strip()).strip(".-")
    if not value:
        raise ValueError("Agent state path component is empty")
    return value


def agent_state_root(repo: Path) -> Path:
    """Return the repository-specific root shared by all agent families."""
    base = Path(
        os.environ.get(
            "METAX_AGENT_HOME",
            os.environ.get("METAX_AGENT_STATE_ROOT", str(DEFAULT_AGENT_HOME)),
        )
    ).expanduser()
    project = os.environ.get("METAX_AGENT_PROJECT", repo.resolve().name)
    return (base / _safe_component(project)).resolve()


def branch_port_home(repo: Path) -> Path:
    """Return the branch-port agent root, separate from upgrade/review agents."""
    override = os.environ.get("METAX_BRANCH_PORT_HOME")
    if override:
        return Path(override).expanduser().resolve()
    return agent_state_root(repo) / "branch-port"


def branch_port_state_dir(repo: Path) -> Path:
    return branch_port_home(repo) / "state"


def branch_port_control_dir(repo: Path) -> Path:
    """Compatibility alias for the former control directory."""
    return branch_port_state_dir(repo)


def branch_port_runs_dir(repo: Path) -> Path:
    return branch_port_home(repo) / "runs"


def branch_port_archive_dir(repo: Path) -> Path:
    return branch_port_home(repo) / "archive"


def branch_port_audits_dir(repo: Path) -> Path:
    """Compatibility alias for archived audits."""
    return branch_port_archive_dir(repo) / "audits"


def branch_port_locks_dir(repo: Path) -> Path:
    return branch_port_state_dir(repo) / "locks"


def default_worktree_root(repo: Path) -> Path:
    return branch_port_home(repo) / "worktrees"


def config_path(repo: Path) -> Path:
    return branch_port_state_dir(repo) / "config.json"


def scope_path(repo: Path) -> Path:
    return branch_port_state_dir(repo) / "scope.json"


def state_path(repo: Path) -> Path:
    return branch_port_state_dir(repo) / "daily-state.json"


def matrix_path(repo: Path) -> Path:
    return branch_port_state_dir(repo) / "matrix.json"


def registry_path(repo: Path) -> Path:
    return branch_port_state_dir(repo) / "worktrees.json"


def policy_root(repo: Path) -> Path:
    """Resolve the checkout that owns the branch-port policy files."""
    override = os.environ.get("METAX_BRANCH_PORT_POLICY_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    repo = repo.resolve()
    marker = repo / ".agents/agents/vllm-metax-branch-port-daily.md"
    if marker.is_file():
        return repo
    sibling = repo.parent / f"{repo.name}-branch-port-policy"
    if (sibling / ".agents/agents/vllm-metax-branch-port-daily.md").is_file():
        return sibling.resolve()
    return repo
