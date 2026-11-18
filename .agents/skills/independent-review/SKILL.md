---
name: independent-review
description: Perform a fresh-context review of staged, unstaged, commit or commit-range code and produce a machine-checkable review artifact bound to the exact reviewed diff.
---

# Independent Review

Use this skill whenever a workflow needs a code review that is independent from the
implementer. It supports:

- staged index review;
- unstaged worktree review, including untracked files;
- one commit against a base;
- a commit range.

This skill is repository-agnostic inside vLLM-MetaX. Domain skills may add their own
applicability, validation or merge gates on top, but should reuse this skill for the
general code-review pass.

## Independence

The reviewer must be a fresh task/subagent or human who did not implement the change.
The implementer cannot write or approve the review artifact.

The reviewer is read-only for source and Git state. It writes only the review JSON.

## Workflow

1. Create a review context bound to the exact target:

   ```bash
   # Staged index
   python .agents/skills/independent-review/scripts/review_context.py snapshot      --repo . --mode staged --output <run-dir>/REVIEW-CONTEXT.json

   # Unstaged worktree, including untracked files
   python .agents/skills/independent-review/scripts/review_context.py snapshot      --repo . --mode worktree --output <run-dir>/REVIEW-CONTEXT.json

   # One commit
   python .agents/skills/independent-review/scripts/review_context.py snapshot      --repo . --mode commit --base <base-sha> --head <commit-sha>      --output <run-dir>/REVIEW-CONTEXT.json

   # Commit range
   python .agents/skills/independent-review/scripts/review_context.py snapshot      --repo . --mode range --base <base-sha> --head <head-sha>      --output <run-dir>/REVIEW-CONTEXT.json
   ```

2. Start a fresh reviewer. Give it the context path, repository/worktree, and the specific
   diff. The reviewer inspects the actual diff and relevant surrounding code.

3. Review correctness, regressions, design fit, error handling, tests, resource lifetime,
   security, compatibility and maintainability. Domain skills may provide additional
   review questions.

4. Write `<run-dir>/REVIEW.json` with the schema in
   [review-modes.md](references/review-modes.md).

5. Validate that the review still belongs to the current target:

   ```bash
   python .agents/skills/independent-review/scripts/check_review.py      --context <run-dir>/REVIEW-CONTEXT.json      --review <run-dir>/REVIEW.json      --verify-context
   ```

For workflows that require approval, add `--require-approval`.

## Output status

- `APPROVE`: no blocking or major findings remain.
- `CHANGES_REQUIRED`: at least one blocking or major finding exists.
- `BLOCKED`: the review target is ambiguous, stale or cannot be inspected.

Do not claim a domain-specific approval from this generic result alone. Branch-port,
model, patch and release workflows must still satisfy their own gates.

## Staleness

`review_context.py snapshot` records hashes of the reviewed diff/index/untracked manifest.
`check_review.py --verify-context` recomputes them. Any changed staged content, worktree
file or commit SHA invalidates the review.
