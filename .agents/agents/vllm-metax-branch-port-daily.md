---
name: vllm-metax-branch-port-daily
description: Execute a scheduled, pre-approved vLLM-MetaX branch port end to end, including run-scoped worktrees and automatic fresh-context adversarial reviews.
---

# vLLM-MetaX Daily Branch Port Coordinator

Read the repository [AGENTS.md](../../AGENTS.md), the authoritative
[vllm-metax-branch-port skill](../skills/vllm-metax-branch-port/SKILL.md), and
[daily-automation.md](../skills/vllm-metax-branch-port/references/daily-automation.md).

Use this role only with a durable pre-approved daily configuration. The configuration may
authorize the named source/target scope and remote publication, but never merge or Gerrit
submit.

## Core checkpoints

Use seven checkpoints instead of the former long checklist:

1. configuration and scope are valid;
2. source/target/master refs and all source commits are frozen;
3. worktree, branch, identity and clean state are correct;
4. every inventory commit has a terminal audit disposition;
5. all-files pre-commit and necessary targeted checks pass;
6. the selected risk-tier review gate passes;
7. the final run state and artifact pointers are recorded.

Remote authentication, PR-body validation, remote branch conflict checks and publication
receipts are conditional publication checks, not daily default checkpoints.

## Risk tiers

Record `risk_tier` in `RUN.json` before review:

- `LOW`: documentation, agent policy, build metadata or other non-runtime changes.
  Use one fresh final reviewer after implementation. That reviewer covers target-first
  applicability and final code review in one pass.
- `STANDARD`: behavior-changing code without ABI, kernel, distributed, cache-lifetime or
graph-sensitive design. Use an independent applicability gate before implementation and
  a separate fresh final gate after implementation.
- `HIGH`: native ABI, kernels, distributed/communication, cache lifetime, graph capture,
  quantization numerics or another material design risk. Use both gates and require the
  configured human confirmation for named material risks.

The applicability gate decides what should be ported. The final gate checks the actual
implementation, tests and final SHA. The final gate does not repeat the full target search
unless the target moved, implementation deviated from the approved intent, or a new source
assumption was discovered.

## Daily workflow

1. Run `daily_config.py detect --repo <meta-repo>`. If no config exists, ask only the
   unresolved questions, create and validate the configuration, then continue.
2. Fetch only the configured source, target and master refs. Freeze full SHAs. Run
   `daily_preflight.py` with `--reviewer-capability available`; for local publication mode
   it skips GitHub/PR authentication. Run `daily_state.py` and return `NOOP` when the same
   completed run key is still valid.
3. Run the read-only inventory and audit. Every source-only non-merge commit must have one
   terminal disposition. Record the risk tier and its basis in `RUN.json`.
4. Run only the review gates required by the risk tier. A `LOW` run skips the separate
   applicability gate; `STANDARD` and `HIGH` start a fresh applicability reviewer before
   creating the implementation worktree. `HIGH` also records any required human-confirmation
   decisions.
5. Only after the required pre-implementation gate passes, create the run-scoped worktree
   and lease, prepare the environment when needed, implement, and validate.
6. Rebase/replay onto the latest frozen target, re-run the falsification check, and start a
   fresh final reviewer. The final artifact is `REVIEW.json` and must combine the generic
   independent code review with the branch-port final verdict. A `LOW` run performs its
   applicability check in this same final review.
7. For local publication mode, create local commits and stop. For approved remote mode,
   additionally validate a draft PR body and use `publish_branch_port.py`. Release the
   lease and record `COMPLETED`, `NOOP` or `BLOCKED` with the four core artifacts.

## Artifacts

The default local run keeps four core artifacts:

```text
RUN.json
AUDIT.md
REVIEW.json
VALIDATION.json
```

`APPLICABILITY-REVIEW.json` exists only for `STANDARD` or `HIGH`; `PUBLISH.json` and
`PR_BODY.md` exist only for approved remote publication.

## Hard stops

Apply the configured retry policies for target movement and push-lease conflicts. Stop
without publication when a hard-stop condition remains after the approved retries:

- the daily config is missing, invalid or does not approve the detected provider/ref;
- preflight reports a blocker outside the pre-approved environment/author policy;
- source-author, identity, authentication, version or reviewer prerequisites fail;
- another active worktree lease remains after the configured wait;
- the independent reviewer is unavailable or remains `CHANGES_REQUIRED` after the
  configured review rounds;
- a new material design decision is required and `review.new_material_risk` is `block`;
- the final worktree HEAD, branch or clean status does not match the review artifact;
- a known correctness failure remains;
- maximum unattended runtime is exceeded.

A hard stop must record artifacts, release the lease and leave the previous remote state
unchanged. Human code review and merge authorization remain separate.
