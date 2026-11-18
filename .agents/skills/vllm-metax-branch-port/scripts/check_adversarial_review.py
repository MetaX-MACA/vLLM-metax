#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Validate an independent adversarial branch-port review artifact."""

from __future__ import annotations

import argparse
import json
import subprocess
import regex as re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
STAGES = {"applicability", "final"}
REVIEWER_MODES = {"independent-agent", "human"}
VERDICTS = {
    "ENDORSE_PORT",
    "ALREADY_COVERED",
    "DO_NOT_PORT",
    "CHANGES_REQUIRED",
}
COMMON_TEXT_FIELDS = (
    "observable_behavior",
    "target_counterpart",
    "target_history_search",
    "target_equivalence_evidence",
    "why_patch_still_needed",
    "falsification_test",
    "expected_failure_if_omitted",
)


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _full_sha(value: Any) -> bool:
    return isinstance(value, str) and SHA_RE.fullmatch(value) is not None


def _git_output(worktree: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(worktree), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ValueError(
            f"git {' '.join(args)} failed in {worktree}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _validate_worktree(
    worktree: Path,
    *,
    expected_head_sha: str | None,
    expected_branch: str | None,
) -> list[str]:
    problems: list[str] = []
    if not worktree.is_dir():
        return [f"review_worktree does not exist: {worktree}"]
    try:
        head_sha = _git_output(worktree, "rev-parse", "HEAD")
        branch = _git_output(worktree, "branch", "--show-current")
        status = _git_output(worktree, "status", "--porcelain")
    except ValueError as error:
        return [str(error)]
    if expected_head_sha is not None and head_sha != expected_head_sha:
        problems.append("review worktree HEAD does not match target_head_sha")
    if expected_branch is not None and branch != expected_branch:
        problems.append("review worktree branch does not match worktree_branch")
    if status:
        problems.append("review worktree must be clean")
    return problems


def validate_review(
    review_path: Path,
    *,
    expected_stage: str | None = None,
    expected_target_sha: str | None = None,
    expected_head_sha: str | None = None,
    expected_worktree: Path | None = None,
    required_intents: Sequence[str] = (),
    require_endorsement: bool = False,
) -> dict[str, Any]:
    data = json.loads(review_path.read_text(encoding="utf-8"))
    problems: list[str] = []

    if data.get("schema_version") != 1:
        problems.append("schema_version must be 1")

    stage = data.get("gate")
    if stage not in STAGES:
        problems.append(f"gate must be one of {sorted(STAGES)}")
    if expected_stage is not None and stage != expected_stage:
        problems.append(f"gate must be {expected_stage}")

    reviewer_mode = data.get("reviewer_mode")
    if reviewer_mode not in REVIEWER_MODES:
        problems.append(f"reviewer_mode must be one of {sorted(REVIEWER_MODES)}")
    for field in (
        "reviewer_identity",
        "implementer_identity",
        "reviewed_at_utc",
        "review_notes",
    ):
        if not _text(data.get(field)):
            problems.append(f"{field} must be non-empty")

    source_shas = data.get("source_shas")
    if (
        not isinstance(source_shas, list)
        or not source_shas
        or any(not _full_sha(sha) for sha in source_shas)
    ):
        problems.append("source_shas must contain full 40-character SHAs")

    target_baseline_sha = data.get("target_baseline_sha")
    if not _full_sha(target_baseline_sha):
        problems.append("target_baseline_sha must be a full 40-character SHA")
    if expected_target_sha is not None and target_baseline_sha != expected_target_sha:
        problems.append("target_baseline_sha does not match the expected target SHA")

    target_head_sha = data.get("target_head_sha")
    if stage == "final" and not _full_sha(target_head_sha):
        problems.append("target_head_sha must be a full SHA at the final gate")
    elif target_head_sha is not None and not _full_sha(target_head_sha):
        problems.append("target_head_sha must be null or a full SHA")
    if expected_head_sha is not None and target_head_sha != expected_head_sha:
        problems.append("target_head_sha does not match the expected head SHA")

    review_worktree = data.get("review_worktree")
    worktree_branch = data.get("worktree_branch")
    if stage == "final":
        if not _text(review_worktree):
            problems.append("review_worktree must be non-empty at the final gate")
        elif not Path(review_worktree).is_absolute():
            problems.append("review_worktree must be an absolute path")
        if not _text(worktree_branch):
            problems.append("worktree_branch must be non-empty at the final gate")
    elif review_worktree is not None and not _text(review_worktree):
        problems.append("review_worktree must be null or a non-empty string")

    if expected_worktree is not None and _text(review_worktree):
        expected_path = expected_worktree.resolve()
        reviewed_path = Path(review_worktree).resolve()
        if reviewed_path != expected_path:
            problems.append("review_worktree does not match the expected worktree")
        else:
            problems.extend(
                _validate_worktree(
                    reviewed_path,
                    expected_head_sha=target_head_sha,
                    expected_branch=worktree_branch,
                )
            )

    intents = data.get("intents")
    if not isinstance(intents, list) or not intents:
        problems.append("intents must be a non-empty list")
        intents = []

    intent_ids: list[str] = []
    verdicts: dict[str, str] = {}
    for index, intent in enumerate(intents):
        prefix = f"intents[{index}]"
        if not isinstance(intent, dict):
            problems.append(f"{prefix} must be an object")
            continue

        intent_id = intent.get("intent_id")
        if not _text(intent_id):
            problems.append(f"{prefix}.intent_id must be non-empty")
        elif intent_id in intent_ids:
            problems.append(f"duplicate intent_id {intent_id}")
        else:
            intent_ids.append(intent_id)

        verdict = intent.get("verdict")
        if verdict not in VERDICTS:
            problems.append(f"{prefix}.verdict must be one of {sorted(VERDICTS)}")
        elif _text(intent_id):
            verdicts[intent_id] = verdict

        for field in COMMON_TEXT_FIELDS:
            if not _text(intent.get(field)):
                problems.append(f"{prefix}.{field} must be non-empty")

        assumptions = intent.get("source_assumptions")
        if (
            not isinstance(assumptions, list)
            or not assumptions
            or any(not _text(item) for item in assumptions)
        ):
            problems.append(
                f"{prefix}.source_assumptions must be a non-empty string list"
            )

        assumptions_valid = intent.get("source_assumptions_still_valid")
        if not isinstance(assumptions_valid, bool):
            problems.append(f"{prefix}.source_assumptions_still_valid must be boolean")
        elif verdict == "ENDORSE_PORT" and not assumptions_valid:
            problems.append(
                f"{prefix} cannot ENDORSE_PORT with invalidated source assumptions"
            )

        risks = intent.get("remaining_risks")
        if not isinstance(risks, list) or any(not _text(item) for item in risks):
            problems.append(f"{prefix}.remaining_risks must be a string list")

        if stage == "final":
            if verdict == "ENDORSE_PORT":
                for field in ("test_fails_without_change", "test_passes_with_change"):
                    if not _text(intent.get(field)):
                        problems.append(
                            f"{prefix}.{field} must be non-empty for ENDORSE_PORT"
                        )

            rebase_sha = intent.get("rebase_reaudit_at_sha")
            if not _full_sha(rebase_sha):
                problems.append(
                    f"{prefix}.rebase_reaudit_at_sha must be a full SHA at final gate"
                )
            elif rebase_sha != target_baseline_sha:
                problems.append(
                    f"{prefix}.rebase_reaudit_at_sha must equal target_baseline_sha"
                )

    missing_intents = sorted(set(required_intents) - set(intent_ids))
    if missing_intents:
        problems.append(f"missing required intents: {', '.join(missing_intents)}")

    blocking = sorted(
        intent_id
        for intent_id, verdict in verdicts.items()
        if verdict != "ENDORSE_PORT"
    )
    if require_endorsement and blocking:
        problems.append(
            "publication requires ENDORSE_PORT for every intent; blocked: "
            + ", ".join(blocking)
        )

    if problems:
        raise ValueError("; ".join(problems))

    return {
        "status": "ok",
        "gate": stage,
        "reviewer_mode": reviewer_mode,
        "source_shas": source_shas,
        "target_baseline_sha": target_baseline_sha,
        "target_head_sha": target_head_sha,
        "review_worktree": review_worktree,
        "worktree_branch": worktree_branch,
        "intents": len(intent_ids),
        "endorsed": sum(verdict == "ENDORSE_PORT" for verdict in verdicts.values()),
        "blocking_verdicts": blocking,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--stage", choices=sorted(STAGES), required=True)
    parser.add_argument("--expected-target-sha")
    parser.add_argument("--expected-head-sha")
    parser.add_argument("--worktree", type=Path)
    parser.add_argument("--intent-id", action="append", default=[])
    parser.add_argument("--require-endorsement", action="store_true")
    args = parser.parse_args(argv)

    try:
        result = validate_review(
            args.review,
            expected_stage=args.stage,
            expected_target_sha=args.expected_target_sha,
            expected_head_sha=args.expected_head_sha,
            expected_worktree=args.worktree,
            required_intents=args.intent_id,
            require_endorsement=args.require_endorsement,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"adversarial review blocked: {error}")
        return 1

    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
