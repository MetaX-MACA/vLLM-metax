# Independent adversarial review

Read this reference when reviewing branch-port applicability or a completed port for
publication. The reviewer is independent from the agent that produced the audit or
implementation. The implementing agent cannot satisfy this gate by reviewing its own work.

## Purpose

The review asks whether a source change should be carried forward at all. Do not begin
from the source diff as the required answer. Begin from the target behavior and try to
falsify the claim that the target still needs a change.

The reviewer must be able to answer:

- what observable behavior is missing on the target;
- which target symbol and call path own that behavior now;
- what target code, history and tests prove the behavior is not already equivalent;
- which source assumptions still hold;
- which validation would fail without the proposed patch and pass with it;
- whether the patch remains necessary after the latest rebase.

A source patch being absent is not evidence of a target defect.

## Independence contract

Use one of these reviewer modes:

- `independent-agent`: a fresh agent/task that did not implement the audit or patch and
  does not inherit the implementer's conclusions as accepted facts;
- `human`: a person who did not author the concrete patch.

The implementer may prepare inputs and respond to findings, but must not write or approve
the independent review artifact. If no independent reviewer is available, record the gate
as pending and do not claim the PR is ready for automatic merge.

The reviewer is read-only for production code and branch state. It may write only its
review artifact under the run directory.

## Locate the review worktree

Do not infer a worktree from a branch name or directory naming convention. Resolve it in
this order:

1. use the coordinator-provided worktree path;
2. read `metax_worktree` from the branch-port `RUN.json` or `worktree` from `PORTS.json`;
3. use `git worktree list --porcelain` and match the exact branch and HEAD;
4. if more than one path remains or the SHA/branch differs, record the gate as blocked.

The final artifact records the absolute `review_worktree` and `worktree_branch`. Before
reviewing, verify:

```bash
git -C <worktree> rev-parse HEAD
git -C <worktree> branch --show-current
git -C <worktree> status --porcelain
```

The HEAD must equal `target_head_sha`, the branch must match `worktree_branch`, and the
worktree must be clean. The publisher additionally requires the reviewed worktree to be
the same checkout it is publishing.

## Gates

### Applicability gate

Run after the commit audit and before user approval of the port implementation. Review
every selected commit or child intent. The artifact is:

```text
.metax-branch-port/runs/<run>/ADVERSARIAL-APPLICABILITY.json
```

The gate must establish target-first evidence and a concrete falsification plan. If the
target already provides an equivalent behavior, recommend `ALREADY_COVERED` rather than
porting source text.

### Final gate

Run after implementation, latest-target rebase, and validation, but before pushing a
Gerrit review or creating/updating a GitHub PR. Review every intent still represented in
the patch. The artifact is:

```text
.metax-branch-port/runs/<run>/ADVERSARIAL-FINAL.json
```

The final gate must include actual fail-before/pass-after behavior evidence, or a narrow
and explicitly approved non-runtime validation contract when execution is impossible.
It must also record the latest target baseline SHA and confirm that the patch was
re-audited after rebasing.

## Generic code review layer

Run the generic [independent-review](../../independent-review/SKILL.md) workflow over
the exact staged, worktree, commit or range target in addition to the branch-port checks
below. Its review context and artifact prevent code from changing between review and use.
A branch-port verdict cannot be `ENDORSE_PORT` while the generic code review is blocking,
stale or absent.

## Required reviewer checks

For each intent:

1. Read the source parent-to-commit diff, not only the final patch summary.
2. Read the target counterpart, its callers, callees and relevant tests.
3. Search target history for equivalent behavior:

   ```bash
   git grep -n <symbol-or-contract>
   git log -S <symbol-or-contract> -- <path>
   git log --follow -- <path>
   git blame <target-file>
   ```

4. Compare input, output, state lifetime, ordering, dispatch, layout, errors and
   performance contract rather than matching code text.
5. Re-check every source assumption against the frozen target revision.
6. Challenge the validation: would it fail if the patch were absent? Does it assert the
   final affected behavior rather than a helper flag or intermediate shape?
7. At the final gate, repeat target equivalence and applicability checks against the
   latest rebased target baseline.

## Verdicts

Use one verdict per intent:

- `ENDORSE_PORT`: target still needs the behavior and the proposed patch is appropriate.
- `ALREADY_COVERED`: target already has equivalent behavior through another mechanism.
- `DO_NOT_PORT`: the intent is obsolete, version-specific or inappropriate for the target.
- `CHANGES_REQUIRED`: the intent may be valid, but evidence, design or validation is not
  sufficient.

Only `ENDORSE_PORT` permits publication. Other verdicts block the affected intent and
must be reflected in the matrix and PR description.

## Artifact schema

Write JSON with this shape:

```json
{
  "schema_version": 1,
  "gate": "applicability",
  "reviewer_mode": "independent-agent",
  "reviewer_identity": "<fresh-task-or-person-id>",
  "implementer_identity": "<audit-or-patch-owner>",
  "reviewed_at_utc": "<timestamp>",
  "source_shas": ["<full-source-sha>"],
  "target_baseline_sha": "<full-target-sha>",
  "target_head_sha": null,
  "review_worktree": null,
  "worktree_branch": null,
  "intents": [
    {
      "intent_id": "<stable-child-intent-id>",
      "verdict": "ENDORSE_PORT",
      "observable_behavior": "<affected behavior>",
      "target_counterpart": "<symbols and call path>",
      "target_history_search": "<searches and result>",
      "target_equivalence_evidence": "<none-found evidence or equivalent mechanism>",
      "source_assumptions": ["<assumption checked on the target>"],
      "source_assumptions_still_valid": true,
      "why_patch_still_needed": "<target-side gap and expected effect>",
      "falsification_test": "<behavior-level test or non-runtime validation contract>",
      "expected_failure_if_omitted": "<observable failure or missing contract>",
      "test_fails_without_change": "<required at final gate>",
      "test_passes_with_change": "<required at final gate>",
      "rebase_reaudit_at_sha": "<required at final gate>",
      "remaining_risks": ["<review risk or approved unverified behavior>"]
    }
  ],
  "review_notes": "<summary and disagreements>"
}
```

At the final gate, `target_head_sha` is the full patched commit, `review_worktree` and
`worktree_branch` identify the clean checkout under review, and `rebase_reaudit_at_sha`
must equal `target_baseline_sha`. The two test-result fields are mandatory for
`ENDORSE_PORT`; blocking verdicts may omit them but cannot permit publication.

## Resolution

- `CHANGES_REQUIRED`: return concrete findings to the implementer, update the audit or
  patch, then request a new independent review.
- `ALREADY_COVERED` or `DO_NOT_PORT`: update the matrix disposition and do not publish
  that intent as a port.
- `ENDORSE_PORT`: record the artifact and continue only after all selected intents have
  an endorsing final review.

The reviewer does not merge, submit or publish the change. Human code review and merge
authorization remain separate.
