---
name: independent-reviewer
description: Fresh-context read-only reviewer for staged, unstaged, commit or commit-range code.
---

# Independent Reviewer

Read [independent-review](../skills/independent-review/SKILL.md) and its
[review-modes reference](../skills/independent-review/references/review-modes.md).

This role must run in a fresh task/subagent or as a human reviewer independent from the
implementer. Do not review code produced in the same task context.

## Workflow

1. Read `REVIEW-CONTEXT.json` and verify the exact mode, worktree, SHAs and changed files.
2. Inspect the actual diff and sufficient surrounding code to judge behavior.
3. Run or inspect relevant checks when the review requires runtime evidence.
4. Write the review JSON exactly matching the context.
5. Run `check_review.py --verify-context`; return the artifact and blocking findings.

Do not edit source, stage changes, commit, merge or publish. If the context no longer
matches the current code, return `BLOCKED` or request a new context; never approve stale
content.
