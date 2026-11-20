# Port scope and validation limits

Read this reference before assigning dispositions, changing the selection or deciding
that missing evidence prevents a selected port from proceeding.

## Keep scope, implementation, validation and review separate

Record scope, implementation and validation independently:

- `selection`: `SELECTED`, `EXCLUDED` or `UNDECIDED`, with the user instruction or
  applicable scope policy that supports it.
- `implementation_status`: `NOT_STARTED`, `REVIEWING`, `IN_PROGRESS`, `COMMITTED`,
  `MERGED` or `BLOCKED`, with a target SHA when one exists.
- Validation checks: name the claim and configuration, result (`PASSED`, `FAILED`,
  `UNVERIFIED` or `NOT_APPLICABLE`), evidence, missing resource and next action.
  Summarize mixed coverage as `PARTIAL`; do not turn an unrun check into a pass.
- Human review: record `review_status`, concrete reasons, the requested user decision
  and the exact confirmation. Migration selection does not itself accept review risks.
- Independent risk-tiered review: record applicability/final artifact paths, reviewer
  mode, per-intent verdicts and the latest rebased target SHA. Pending or non-endorsing
  review blocks automatic merge and publication.

The disposition describes the migration decision. A user-required capability missing
from the target is `PORT_REQUIRED`, even when implementation review or runtime
validation is pending. Missing evidence does not revoke `selection: SELECTED`.

Use `NEEDS_REVIEW` when applicability or the requested scope is genuinely undecided.
For a selected port with unresolved implementation details, keep the requirement and
record `implementation_status: REVIEWING` instead of silently deselecting it.

## Scope follows the current request

Carry relevant source behavior forward when the user asks to port a branch. Ordinary
feature work is not excluded merely because it is not a bug fix. Independently account
for documentation, formatting and test changes under the user's contribution policy;
they do not justify dropping an associated production capability.

`OUT_OF_SCOPE` requires an explicit user exclusion or a concrete, approved scope rule.
Record its basis. Missing GPUs, FP8 support, model weights, dependencies, traces or
benchmarks are validation limits, never scope exclusions. A merge conflict or a larger
adaptation effort is not an exclusion either.

A later instruction to include previously excluded work supersedes that exclusion.
Retain the old decision as history, reopen its commit and child intents, and update
`AUDIT.md`, decisions, run state and matrix. Preserve already-merged child mappings.
Do not reuse the old exclusion as a new permission gate.

A full-branch port still avoids duplicate or dead behavior. `ALREADY_PRESENT`,
`UPSTREAM_FIXED`, `VERSION_SPECIFIC` and `OBSOLETE` need positive target evidence;
they cannot be inferred from unavailable validation. Split a mixed commit into explicit
child intents under the approved grouping policy so partial delivery stays visible.

## Define what the evidence must establish

| Question | Evidence to seek | If evidence is missing |
| --- | --- | --- |
| Is the source capability absent or different on the target? | Inspect source and target entry points, dispatch conditions, registration, contracts and an explanatory scenario. | Continue call-path analysis. Keep a user-selected port required; list the unresolved implementation question. |
| Does an existing implementation make the port unnecessary? | Identify the exact target symbols or replacement revision and show equivalent behavior over the relevant conditions. | Do not mark it already present, obsolete or version-specific. |
| Can the adaptation be implemented coherently? | Resolve API/layout changes, ownership, dependency order, state lifetime and stream/event invariants. | Investigate code and feasible checks. Block only the affected implementation step if a concrete unresolved contract prevents a sound patch. |
| Is correctness validated on a particular configuration? | Run the relevant component, independent reference or integration check on that configuration. | Mark that check `UNVERIFIED` in private run state, retain the missing-resource detail there and run other meaningful checks. Static review does not count as numerical validation. |
| Is performance improved on the target? | Paired measurements or traces with the same workload, hardware and configuration. | Mark target benefit unmeasured. Port the selected optimization without claiming a speedup; a missing benchmark alone is not a correctness failure. |

Do not require the user to supply evidence that the agent can investigate. State the
specific symbol, invariant, configuration or decision that remains unresolved and the
next feasible action; avoid a bare “insufficient evidence” verdict. Keep missing-resource
reasons and vendor/chip or private environment details in private run state. A public PR
lists only `- Pending verification: <affected behavior or claim>` for the pending check.

## Proceed with bounded validation

Run feasible static checks, imports, builds and focused component checks. If FP8
execution needs unavailable hardware or a dependency is absent from the fixed approved
environment, preserve the source capability and adapt its target integration. Record
exactly which execution was not tested and why in private run state. Do not remove the
path, replace it with a no-op, or change its default merely to avoid that validation gap.

Known incorrect results, broken ordering, a failed affected check or an unexplained
ABI mismatch are implementation problems to fix. They are not waived by labeling the
work unverified. An external resource gap may prevent a particular test while other
implementation and validation work continues. Ask only for a material decision or
resource that cannot be inferred or obtained within the authorized task.

Correctness and performance claims are separate. A benchmark is required to claim a
target speedup; it is not automatically required to carry forward every performance
change. If timing is itself part of correctness, such as a race or buffer lifetime,
validate that invariant independently of the speedup measurement.

When source comments or a subject conflict with executed code, trace the actual source
behavior, callers and history. Preserve the executed behavior by default and document
the inconsistency. Correct a demonstrated defect as part of adaptation and explain the
deviation. Do not automatically make an optimization opt-in, disable it, or demand a
new product approval just because its subject says “opt-in”. If equally plausible
contracts remain, ask the smallest concrete question while continuing independent work.

## Independent adversarial review

Use [adversarial-review.md](adversarial-review.md) and the portable
[branch-port reviewer](../../../agents/vllm-metax-branch-port-reviewer.md). The reviewer
must be a fresh agent/task or human who did not implement the patch. The implementer can
prepare evidence but cannot write the approval artifact.

Use one fresh final review for `LOW` risk. Use `APPLICABILITY-REVIEW.json` before
implementation and a different fresh final `REVIEW.json` for `STANDARD`/`HIGH`. Every
selected intent needs `ENDORSE_PORT`; `ALREADY_COVERED`, `DO_NOT_PORT`, `CHANGES_REQUIRED`
and a missing required review remain blockers. A plan confirmation or PR-publication
approval is not an independent review result.

## Explicit confirmation for items needing review

When a commit or child intent needs human review, present it to the user explicitly.
Do not leave the reason only in a JSON file or a generic “needs review” summary. Keep
`selection: SELECTED` and its migration requirement; track this gate independently.

For each review item, provide a compact confirmation card:

- source SHA or child intent, proposed behavior and source/target baseline;
- the explicit statement **“暂不满足自动合并”** (not currently eligible for automatic
  merge), followed by concrete reasons;
- what is known, what remains unverified or disputed, and the affected configuration;
- the implementation/validation action that can resolve each reason;
- the exact decision being requested, such as a default, compatibility boundary,
  adaptation approach or acceptance of a specifically named untested configuration;
- the effect of confirmation: continue the named approach or publish a review with
  the disclosed limit. Do not interpret it as approval to merge.

Examples of concrete reasons include a comment/code default conflict, an unresolved
stream or buffer-lifetime contract, a Torch ABI boundary, or unexecuted FP8/device
checks. A missing benchmark means target benefit is unmeasured; say this specifically
rather than claiming a correctness failure. If several child intents share a single
review decision, group the card but retain their source-to-child mapping.

Set `review_status: AWAITING_USER_CONFIRMATION` and
`automatic_merge_eligible: false` after presenting the card. Keep the pending question
visible in the handoff and record the user's explicit response when received. A prior
instruction to migrate the source, or approval to publish a PR, is not an answer to a
newly identified material review question. Reuse an earlier explicit confirmation of
the same risk and proposal; do not request it again without a material change.

Before the dependent action, obtain the named confirmation. Meanwhile continue
independent analysis, implementation preparation and feasible checks. The absence of
an answer is not consent, but it also does not turn the item into `OUT_OF_SCOPE`.
If there is no reviewable patch yet, label the card as confirmation of the approach;
request final code review when the concrete patch and checks are ready.

Confirmation records must bind the decision to source/target SHAs and the proposed
artifact or patch SHA when available. Record whether the user accepted an unverified
configuration, selected an implementation choice, deferred the decision or requested
changes. Confirmation does not convert an unrun check to `PASSED`, fix a known defect,
or prove a performance gain. Resolve implementation failures before calling them done.

The workflow does not automatically submit Gerrit changes or merge GitHub PRs, even
after this confirmation. Report review readiness and provider checks separately;
merging requires a separate explicit instruction and the applicable checks.

## Reporting and completion

A target commit can be `PORTED` when the required behavior is implemented, feasible
checks pass and no known implementation failure remains. Keep unexecuted hardware or
model coverage explicitly `UNVERIFIED`; this is not a fully validated configuration.
Honor any specifically required acceptance test instead of silently waiving it.

Publishing a reviewable port follows the user's existing publication authorization,
required checks and the explicit confirmations above for review items. Reuse already
confirmed limits; do not treat scope approval as confirmation of a new review risk.
An open review does not authorize merging it.

Edge closure requires all selected behavior to be delivered and all required checks
to be resolved. An accepted external coverage limit is recorded separately and cannot
be described as a pass. If such a limit prevents the promised acceptance claim, keep
that claim or edge open without abandoning the port. Never close an edge by recasting
unfinished selected work as `OUT_OF_SCOPE`.

Keep private provenance separate from public review text when the user restricts
source identity disclosure. Follow the user's README, test-file and commit/PR grouping
instructions when packaging the work; local validation does not imply permission to
include its test files in a commit.
