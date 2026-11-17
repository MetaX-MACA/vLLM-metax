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
[commit-logic-audit.md](commit-logic-audit.md). That section must explain the execution
logic and provide a simple user-facing explanation example. The example is not required
to include numerical validation or a reproduced test result. A row without this section
is not eligible for approval as a port candidate.

A target commit with the same patch ID is strong evidence that the exact change is
already present. It is not sufficient by itself: target code may contain an equivalent
or incomplete adaptation under different text. Conversely, a missing patch ID does not
prove that the behavior is needed.

## Decisions

Use one decision for each source commit or tightly coupled Gerrit change:

| Decision | Meaning | Required evidence |
| --- | --- | --- |
| `PORT_REQUIRED` | The target still has the defect, performance problem or missing required adaptation addressed by the source change. | Reproducible target code path, test or precise behavioral comparison, plus why the source approach applies after adaptation. |
| `NEEDS_REVIEW` | Evidence is incomplete, conflicting, spans unrelated intentions, or depends on an unresolved product decision. | State the concrete missing evidence and the exact reviewer decision required. |
| `ALREADY_PRESENT` | The target already contains the same semantic change, including an equivalent adaptation. | Target symbol or test showing current behavior; patch ID is supporting evidence only. |
| `UPSTREAM_FIXED` | Target upstream or another target change fixes the same problem differently. | Link or identify the replacement revision, symbol and behavior; explain why the old source change is unnecessary. |
| `VERSION_SPECIFIC` | The changed behavior intentionally applies only to the source version. | Evidence such as removed feature/API contract, unsupported model or chip scope, or an explicit version policy. A cherry-pick conflict is not evidence. |
| `OBSOLETE` | The affected code path or feature no longer exists and has no target replacement requiring the change. | Target-wide reference search and call-path analysis, not only an absent file. |
| `OUT_OF_SCOPE` | The commit is documentation, formatting, test-only, ordinary feature work or otherwise outside the approved port policy. | State the category and any user override. |

When evidence does not meet the bar, use `NEEDS_REVIEW`. Do not turn uncertainty into
`VERSION_SPECIFIC` or `OBSOLETE`.

## Terminal dispositions and edge closure

The linear matrix requires complete accounting, not literal commit ancestry.

- `PORT_REQUIRED` becomes `PORTED` only after the target commit exists and its
  validation evidence is current.
- `NEEDS_REVIEW` remains nonterminal until the user or required reviewer supplies a
  decision.
- `ALREADY_PRESENT`, `UPSTREAM_FIXED`, `VERSION_SPECIFIC` and `OBSOLETE` are terminal
  when their required evidence is recorded.
- `OUT_OF_SCOPE` is terminal only under the approved scope policy.

Do not close an edge while any `PORT_REQUIRED`, `PORTED` without a target SHA,
`NEEDS_REVIEW` or stale-baseline item remains. A terminal non-port disposition creates
no target commit, so the source commit does not appear in the next downstream audit.

## What still needs porting

A change remains a candidate when any of the following is true on the target:

- the same failure, crash, hang, wrong result or compatibility defect remains;
- the same performance bottleneck remains on a relevant supported configuration;
- a required model, quantization, attention or kernel adaptation is still absent;
- a build, dependency, security or packaging fix is required by the target environment;
- a target-specific refactor replaced the original implementation, but its semantics
  must retain the source fix.

For performance changes, identify the affected workload, hardware and configuration.
If the bottleneck cannot be established statically, mark the benchmark or reproducer
needed before committing. A historical measured gain on another version is not proof
of a target gain.

For model adaptations, verify that the model is in target scope, its registration and
model-owned dependency chain still apply, and the source implementation does not rely
on removed framework contracts.

## Commit grouping

Treat a Gerrit merge wrapper as packaging, not as the unit of work. Use its underlying
non-merge change commit. If several commits share one original `Change-Id` and cannot
be understood independently, use that logical group for the audit decision. The approved
port still produces one target commit per selected source commit, unless the user
explicitly approves a different mapping.

If one source commit contains unrelated intentions, mark it `NEEDS_REVIEW`; do not
silently split or combine it.

## Report template

```markdown
# Branch port audit: <source> -> <target>

- Remote: `<remote>`
- Fetched at: `<UTC timestamp>`
- Source: `<ref>` @ `<full-sha>`
- Target: `<ref>` @ `<full-sha>`
- Target vLLM: `<ref>` @ `<full-sha>` / `<public-version>`
- Target MetaX public version: `<public-version>` / `<matches-vllm: yes-or-no>`
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

| Source | Author | Intent | Files/domain | Target evidence | Migration/adaptation required | Decision | Proposed commit | Depends on | Validation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

Every inventory commit must have exactly one row. Do not group commits or move
already-adapted commits to an unlisted summary.

## Commit logic audits

<One detailed section per PORT_REQUIRED commit, using
references/commit-logic-audit.md. Each section must include an explanation example.>

## Ignored merge wrappers

<List source-only merge commits and the change commits they wrap.>

## Proposed port order

1. `<source-sha>` - `<reason and dependency>`

## Approval needed

- Selected commits:
- Original author identities (`Name <email>`):
- Porting developer name/email:
- Source-author confirmation:
- Target ref and frozen SHA:
- Target vLLM mapping and public version:
- MetaX/vLLM public-version equality:
```
