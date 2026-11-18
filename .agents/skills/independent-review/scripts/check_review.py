#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Validate a generic independent code-review artifact against its context."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from review_context import verify_context

REVIEWER_MODES = {"independent-agent", "human"}
STATUSES = {"APPROVE", "CHANGES_REQUIRED", "BLOCKED"}
SEVERITIES = {"blocking", "major", "minor", "info"}
CHECK_STATUSES = {"PASSED", "FAILED", "NOT_RUN", "NOT_APPLICABLE"}


def text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def validate_review(
    context_path: Path,
    review_path: Path,
    *,
    verify_current_context: bool = False,
    require_approval: bool = False,
) -> dict[str, Any]:
    context_bytes = context_path.read_bytes()
    context = json.loads(context_bytes)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    problems: list[str] = []

    if context.get("schema_version") != 1 or review.get("schema_version") != 1:
        problems.append("schema_version must be 1 for context and review")
    if review.get("review_mode") != context.get("review_mode"):
        problems.append("review_mode does not match the context")
    if review.get("review_hash") != context.get("review_hash"):
        problems.append("review_hash does not match the context")
    context_sha256 = hashlib.sha256(context_bytes).hexdigest()
    if review.get("review_context_sha256") != context_sha256:
        problems.append("review_context_sha256 does not match the context file")

    if review.get("reviewer_mode") not in REVIEWER_MODES:
        problems.append(f"reviewer_mode must be one of {sorted(REVIEWER_MODES)}")
    for field in (
        "reviewer_identity",
        "implementer_identity",
        "reviewed_at_utc",
        "summary",
    ):
        if not text(review.get(field)):
            problems.append(f"{field} must be non-empty")

    status = review.get("status")
    if status not in STATUSES:
        problems.append(f"status must be one of {sorted(STATUSES)}")

    findings = review.get("findings")
    if not isinstance(findings, list):
        problems.append("findings must be a list")
        findings = []
    blocking_severities: list[str] = []
    for index, finding in enumerate(findings):
        prefix = f"findings[{index}]"
        if not isinstance(finding, dict):
            problems.append(f"{prefix} must be an object")
            continue
        severity = finding.get("severity")
        if severity not in SEVERITIES:
            problems.append(f"{prefix}.severity must be one of {sorted(SEVERITIES)}")
        elif severity in {"blocking", "major"}:
            blocking_severities.append(severity)
        for field in ("path", "summary", "evidence", "required_action"):
            if not text(finding.get(field)):
                problems.append(f"{prefix}.{field} must be non-empty")
        line = finding.get("line")
        if line is not None and (type(line) is not int or line <= 0):
            problems.append(f"{prefix}.line must be a positive integer or null")

    checks = review.get("checks")
    if not isinstance(checks, list):
        problems.append("checks must be a list")
        checks = []
    elif status != "BLOCKED" and not checks:
        problems.append("non-blocked reviews require at least one check")
    for index, check in enumerate(checks):
        prefix = f"checks[{index}]"
        if not isinstance(check, dict):
            problems.append(f"{prefix} must be an object")
            continue
        if not text(check.get("name")):
            problems.append(f"{prefix}.name must be non-empty")
        if check.get("status") not in CHECK_STATUSES:
            problems.append(f"{prefix}.status must be one of {sorted(CHECK_STATUSES)}")
        if not text(check.get("evidence")):
            problems.append(f"{prefix}.evidence must be non-empty")

    if status == "APPROVE" and blocking_severities:
        problems.append("APPROVE cannot contain blocking or major findings")
    if status == "CHANGES_REQUIRED" and not blocking_severities:
        problems.append("CHANGES_REQUIRED requires a blocking or major finding")
    if require_approval and status != "APPROVE":
        problems.append("approval is required")

    if verify_current_context:
        try:
            verify_context(context)
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            problems.append(str(error))

    if problems:
        raise ValueError("; ".join(problems))

    return {
        "status": "ok",
        "review_status": status,
        "review_mode": context["review_mode"],
        "changed_files": len(context.get("changed_files", [])),
        "findings": len(findings),
        "blocking_findings": len(blocking_severities),
        "review_hash": context["review_hash"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--verify-context", action="store_true")
    parser.add_argument("--require-approval", action="store_true")
    args = parser.parse_args(argv)

    try:
        result = validate_review(
            args.context,
            args.review,
            verify_current_context=args.verify_context,
            require_approval=args.require_approval,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"independent review blocked: {error}")
        return 1

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
