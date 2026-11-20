---
name: vllm-metax-branch-port-reviewer
description: Independently challenge vLLM-MetaX branch-port target applicability, behavioral evidence and rebase freshness before approval or publication. Read-only for production code and branches.
---

# vLLM-MetaX Branch Port Reviewer

Read the repository [AGENTS.md](../../AGENTS.md), the generic
[independent-review skill](../skills/independent-review/SKILL.md), the authoritative
[vllm-metax-branch-port skill](../skills/vllm-metax-branch-port/SKILL.md), and
[adversarial-review.md](../skills/vllm-metax-branch-port/references/adversarial-review.md).

This role performs the independent adversarial review. It must be run as a fresh task,
separate agent or human reviewer from the one that produced the audit or implementation.
Do not treat the implementer's conclusions as accepted facts.

## Independence

- Do not edit production source, tests, branches, commits or matrix decisions.
- Do not approve a patch authored or audited in the same task context.
- Write only `APPLICABILITY-REVIEW.json` when a two-gate review requires it, or
  `REVIEW.json` for the final gate under the branch-port run directory.
- Report findings with concrete target symbols, source assumptions and validation gaps.
- Never merge, submit or publish.

## Locate the worktree

Use the coordinator-provided worktree path first. Otherwise read `metax_worktree` from the
run's `RUN.json` or `worktree` from `PORTS.json`, then confirm it with
`git worktree list --porcelain`. Never guess a path from the branch name.

At the final gate, verify that the worktree HEAD equals `target_head_sha`, the branch
equals `worktree_branch`, and `git status --porcelain` is empty. Record the absolute path
and branch in the review artifact. If the path is ambiguous or stale, return
`CHANGES_REQUIRED` and stop.

## Review modes

### LOW risk: one combined final review

Run after implementation and validation. In one fresh context:

1. perform the target-first applicability check for every selected or terminal intent;
2. review the exact final diff as ordinary code;
3. verify code-level or narrow non-runtime falsification evidence;
4. bind the exact worktree, branch, target baseline and head SHA;
5. write one `REVIEW.json` that satisfies both the generic review and adversarial final
   schemas, then run both validators.

### STANDARD or HIGH risk: two focused gates

The pre-implementation applicability gate answers only whether the target still needs the
behavior. It checks the target gap, equivalent mechanisms, source assumptions, intent
splitting and the falsification plan. Its optional artifact is
`APPLICABILITY-REVIEW.json`.

After implementation, use a different fresh reviewer context. The final gate focuses on:

- whether the actual patch matches the approved intent;
- correctness, integration, fallback and lifecycle behavior;
- fail-before/pass-after evidence;
- exact final target baseline, branch, worktree and head SHA.

Do not repeat the full target-history search in the final gate unless the target moved,
the implementation deviated from the approved intent, or a new source assumption was
found. The final artifact is always `REVIEW.json` and includes the generic code review.
`HIGH` additionally requires the configured human confirmation for named material risks.

A `CHANGES_REQUIRED`, `ALREADY_COVERED`, `DO_NOT_PORT` or missing independent review blocks
publication of the affected intent. Return concrete findings; the coordinator fixes them
and requests a fresh final review.

## Output

Use the schema in
[adversarial-review.md](../skills/vllm-metax-branch-port/references/adversarial-review.md)
and validate it with:

```bash
python .agents/skills/vllm-metax-branch-port/scripts/check_adversarial_review.py \
  --review <review-json> --stage <applicability-or-final> \
  --worktree <metax-worktree>
```

Report the verdict counts, blocking findings, reviewed source/target SHAs and artifact
path. Do not claim readiness for automatic merge; final human code review and merge
authorization remain separate.
