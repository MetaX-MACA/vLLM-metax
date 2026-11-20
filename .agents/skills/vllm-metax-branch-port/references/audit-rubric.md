# Cross-branch audit rubric

The source is the older development branch and the target is the newer branch receiving
carry-forward work. The user may override this orientation only by explicitly naming
which branch is the source.

## Candidate universe

Before creating the candidate universe, fetch the exact source and target branches and
freeze their full SHAs from the refreshed remote-tracking refs. Never audit a stale
local or remote-tracking branch merely because it already exists in the checkout.

Start with commits reachable from the frozen source SHA but not the frozen target SHA.
Exclude merge wrapper commits and inspect their underlying non-merge commits. Keep all
excluded merge wrappers in the report for traceability.

The audit is an exhaustive accounting of the inventory, not a list of suspected ports.
Every non-merge commit must produce its own report row, including commits that appear
already adapted, superseded, version-specific or otherwise unlikely to need migration.
Do not pre-filter using patch IDs, subjects, file names or previous oral conclusions.

For every candidate, capture:

- full SHA, parents, author and committer names and email addresses;
- subject and full commit message, including issue IDs;
- original `Change-Id`, used only as metadata;
- changed files and a stable patch ID where available;
- relevant target files, symbols, tests and upstream comparisons.

Every row must state:

- the target-side evidence inspected;
- whether the target still needs the source behavior;
- whether migration or target-specific adaptation is required;
- the resulting disposition and any missing evidence.

Every commit marked `PORT_REQUIRED` must also have a detailed section following
[commit-logic-audit.md](commit-logic-audit.md). That section must include target-first
applicability proof, checked source assumptions, a behavior-level falsification plan,
the execution logic and a simple user-facing explanation example. A missing source patch
is not target evidence. A row without this section is not eligible for approval as a port
candidate.

A selected commit or child intent additionally requires independent adversarial review at
the applicability gate. The reviewer must actively search for an equivalent target
mechanism and try to show that the port is unnecessary. The implementing agent cannot
self-approve this gate.

A target commit with the same patch ID is strong evidence that the exact change is
already present. It is not sufficient by itself: target code may contain an equivalent
or incomplete adaptation under different text. Conversely, a missing patch ID does not
prove that the behavior is needed.

## Decisions

Apply [scope-and-validation.md](scope-and-validation.md). Keep the user's selection,
implementation progress and per-configuration validation results independent.
A selected missing capability remains required while open implementation questions
are investigated or runtime checks are unavailable.

Use one decision for each source commit or tightly coupled Gerrit change:

| Decision | Meaning | Required evidence |
| --- | --- | --- |
| `PORT_REQUIRED` | Source behavior requested by the user is absent or needs adaptation on the target. | Prove target-first applicability, search target code/history for an equivalent mechanism, validate source assumptions and define a behavior-level falsification check. Record implementation questions and validation limits separately. A reproduced performance gain is not required to select the port. |
| `NEEDS_REVIEW` | Applicability or scope is genuinely undecided. | State the exact unresolved contract or decision and investigate it. Do not deselect user-required work or use this merely because runtime validation is unavailable. |
| `ALREADY_PRESENT` | The target already contains the same semantic change, including an equivalent adaptation. | Target symbol or test showing current behavior; patch ID is supporting evidence only. |
| `UPSTREAM_FIXED` | Target upstream or another target change fixes the same problem differently. | Link or identify the replacement revision, symbol and behavior; explain why the old source change is unnecessary. |
| `VERSION_SPECIFIC` | The changed behavior intentionally applies only to the source version. | Evidence such as removed feature/API contract, unsupported model or chip scope, or an explicit version policy. A cherry-pick conflict is not evidence. |
| `OBSOLETE` | The affected code path or feature no longer exists and has no target replacement requiring the change. | Target-wide reference search and call-path analysis, not only an absent file. |
| `OUT_OF_SCOPE` | The user or an approved scope policy explicitly excludes this work. | Record the exclusion basis. Missing hardware, FP8 validation, model weights or performance evidence never establishes an exclusion. A later inclusion instruction supersedes it. |

When target applicability is uncertain, investigate and record `NEEDS_REVIEW` without
losing the user's selection. For user-required work, keep `PORT_REQUIRED` and record
pending implementation review. Do not turn uncertainty into a terminal non-port
disposition or treat an unrun check as evidence that the capability is unnecessary.

## Terminal dispositions and edge closure

The linear matrix requires complete accounting, not literal commit ancestry.

- `PORT_REQUIRED` becomes `PORTED` after the target commit implements the selected
  behavior, feasible checks pass and no known implementation failure remains. Record
  external coverage limits separately; specifically required acceptance checks remain
  required.
- `NEEDS_REVIEW` remains nonterminal while applicability or scope is unresolved. The
  agent should resolve code questions within the authorized task instead of always
  waiting for the user to supply evidence.
- `ALREADY_PRESENT`, `UPSTREAM_FIXED`, `VERSION_SPECIFIC` and `OBSOLETE` are terminal
  when their required evidence is recorded.
- `OUT_OF_SCOPE` is terminal only under the approved scope policy.

Do not close an edge while any `PORT_REQUIRED`, `PORTED` without a target SHA,
`NEEDS_REVIEW` or stale-baseline item remains. A terminal non-port disposition creates
no target commit, so the source commit does not appear in the next downstream audit.

## Human review confirmation

For each item requiring human review, add the explicit confirmation card from
[scope-and-validation.md](scope-and-validation.md#explicit-confirmation-for-items-needing-review).
State “暂不满足自动合并”, concrete reasons, affected configurations, resolving actions
and the exact user decision. Record pending/confirmed status and the response, keeping
the item's migration requirement unchanged. Do not count selection as risk acceptance.

## What still needs porting

A change remains a candidate when any of the following is true on the target:

- the same failure, crash, hang, wrong result or compatibility defect remains;
- the same performance bottleneck remains on a relevant supported configuration;
- a required model, quantization, attention or kernel adaptation is still absent;
- a build, dependency, security or packaging fix is required by the target environment;
- a target-specific refactor replaced the original implementation, but its semantics
  must retain the source fix.

For performance changes, identify the affected workload, hardware and configuration.
If target benefit cannot be established statically, record a benchmark or reproducer
as needed to substantiate the performance claim. Continue a selected semantic port
with feasible correctness checks and label the target gain unmeasured. A historical
gain on another version is not proof of a target gain; unavailable measurements alone
do not prohibit committing a sound adaptation.

For model adaptations, verify that the model is in target scope, its registration and
model-owned dependency chain still apply, and the source implementation does not rely
on removed framework contracts.

## Independent adversarial review

For every selected commit or child intent, run the generic
[independent reviewer](../../../agents/independent-reviewer.md) with the
[branch-port review profile](reviewer-profile.md) using
[adversarial-review.md](adversarial-review.md) before the port approval and again before
publication. The applicability artifact is `APPLICABILITY-REVIEW.json`; the
post-rebase artifact is `REVIEW.json`. Only `ENDORSE_PORT` permits the affected
intent to be published. `ALREADY_COVERED`, `DO_NOT_PORT`, `CHANGES_REQUIRED` and a missing
review are blockers, not implicit approvals.

## Commit grouping

Treat a Gerrit merge wrapper as packaging, not as the unit of work. Use its underlying
non-merge change commit. If several commits share one original `Change-Id` and cannot
be understood independently, use that logical group for the audit decision. The approved
port still produces one target commit per selected source commit, unless the user
explicitly approves a different mapping.

If one source commit contains unrelated intentions, enumerate its child intents and
preserve the requested scope. Follow the approved split/grouping policy; do not silently
split, combine or mark a partly delivered commit fully ported. Review questions about
one child do not exclude its siblings.

## Report template

```markdown
# Branch port audit: <source> -> <target>

- Remote: `<remote>`
- Fetched at: `<UTC timestamp>`
- Source: `<ref>` @ `<full-sha>`
- Target: `<ref>` @ `<full-sha>`
- Requested source and final target: `<source> -> <final-target>`
- Resolved branch chain: `<ref -> ref -> ...>`
- Skipped `mx/` branches: `<branches-or-none>`
- Source remote: `<logical-name>` / `<provider>`
- Target remote: `<logical-name-or-none>` / `<provider>`
- Target vLLM: `<ref>` @ `<full-sha>` / `<public-version>`
- Target MetaX public version: `<public-version>` / `<matches-vllm: yes-or-no>`
- Final target: `master`
- Latest `mx/vX.Y.Z-dev` public version: `<version>`
- GitHub master public version: `<version>` / `<master-version-gate: pass-or-blocked>`
- Remote publication requested: `<yes-or-no>`
- Publication provider: `<gerrit/github/local>`
- Publication remote: `<logical-name-or-none>`
- Gerrit review ref or GitHub PR plan: `<ref-or-base/head-plan>`
- Master local-only reason: `<reason-or-none>`
- Master commit identity policy: `<porting-developer-as-author-and-committer/no-source-trailer>`
- Matrix edge: `<edge-id>`
- Predecessor edge: `<edge-id or none>` / `<status>`
- Edge status: `<OPEN/IN_PROGRESS/BLOCKED/CLOSED>`
- Merge base: `<full-sha>`
- Candidate commits: `<count>`
- Recommended ports: `<count>`
- Needs review: `<count>`
- Inventory non-merge commits: `<count>`
- Report commit rows: `<count>`
- Completeness: `<counts-equal: yes-or-no>`

## Summary

<Short explanation of the compatibility interval and main risks.>

## Complete commit decisions

| Source | Intent | Selection and basis | Target evidence | Decision | Implementation status | Proposed commit | Depends on | Validation results and limits |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

Every inventory commit must have exactly one row. Do not group commits or move
already-adapted commits to an unlisted summary.

## Commit logic audits

<One detailed section per PORT_REQUIRED commit, using
references/commit-logic-audit.md. Each section must include an explanation example.>

## Ignored merge wrappers

<List source-only merge commits and the change commits they wrap.>

## Validation limits

<List each unrun check, its exact configuration, missing resource and next action.
Distinguish unavailable checks from known failures and unmeasured performance claims.
Keep private author provenance outside public reports when required by the user.>

## Review confirmation requests

<For each item: source/child intent, concrete proposal, “暂不满足自动合并”, reasons,
known results versus limits, resolving actions, exact user decision and pending or
confirmed response. A plan confirmation is not final code review or merge approval.>

## Proposed port order

1. `<source-sha>` - `<reason and dependency>`

## Approval needed

- Selected commits:
- Confirmed source-to-final-target branch matrix:
- Publish to remote: `<yes-or-no>`
- Publication provider and logical remote:
- Gerrit review ref or GitHub PR base/head:
- Final target and master version gate:
- Master local-only reason:
- Original author identities (`Name <email>`):
- Porting developer name/email:
- Source-author confirmation:
- Target ref and frozen SHA:
- Target vLLM mapping and public version:
- MetaX/vLLM public-version equality:
- Master commit identity policy:
```
