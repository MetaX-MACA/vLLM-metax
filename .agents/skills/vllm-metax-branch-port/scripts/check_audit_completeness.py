#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Verify every inventory commit appears in a branch-port audit report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections.abc import Sequence


def missing_commits(inventory_path: Path, report_path: Path) -> list[dict[str, str]]:
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    report = report_path.read_text(encoding="utf-8")
    commits = inventory.get("commits")
    if not isinstance(commits, list):
        raise ValueError("inventory must contain a commits list")

    missing = []
    for commit in commits:
        sha = commit.get("sha")
        if not isinstance(sha, str) or len(sha) < 40:
            raise ValueError("inventory commit is missing a full SHA")
        if sha not in report:
            missing.append({"sha": sha, "subject": str(commit.get("subject", ""))})
    return missing


def invalid_port_audits(report_path: Path, port_shas: Sequence[str]) -> list[str]:
    report = report_path.read_text(encoding="utf-8")
    invalid = []
    for sha in port_shas:
        marker = f"## Commit logic audit: {sha}"
        start = report.find(marker)
        if start < 0:
            invalid.append(f"{sha}: missing logic-audit section")
            continue
        next_section = report.find("\n## ", start + len(marker))
        section = report[start : next_section if next_section >= 0 else None]
        if "### Explanation example" not in section:
            invalid.append(f"{sha}: missing explanation example")
    return invalid


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--port-sha",
        action="append",
        default=[],
        help="PORT_REQUIRED source SHA; repeat for every selected port candidate",
    )
    args = parser.parse_args(argv)

    missing = missing_commits(args.inventory, args.report)
    if missing:
        for commit in missing:
            print(f"MISSING {commit['sha']} {commit['subject']}")
        print(f"Audit is incomplete: {len(missing)} commit(s) missing")
        return 1

    invalid = invalid_port_audits(args.report, args.port_sha)
    if invalid:
        for issue in invalid:
            print(f"INCOMPLETE PORT AUDIT {issue}")
        return 1

    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    print(
        f"Audit is complete: {inventory['commit_count']} commit(s) present; "
        f"{len(args.port_sha)} port logic audit(s) validated"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
