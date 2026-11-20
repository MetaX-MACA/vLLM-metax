# Daily branch-port automation

Use this reference for a scheduled, pre-approved end-to-end branch port. The human approves
a bounded scope and publication policy once in a durable configuration; each daily run is
then idempotent, run-scoped and blocked only by changed facts or unresolved risk.

## One daily entry point

Start one fresh Codex task from the repository root:

```text
Read .agents/agents/vllm-metax-branch-port.md and execute the daily port in daily mode
using $METAX_BRANCH_PORT_HOME/state/config.json. Do not request approval again for facts
already covered by that configuration. Do not merge or submit.
```

The coordinator starts only the reviewer gates required by the recorded risk tier. `LOW`
runs use one fresh final reviewer; `STANDARD` and `HIGH` use focused applicability and
final reviewers in separate fresh contexts.

## Bootstrap and auto-detection

Before requiring a configuration file, run:

```bash
python .agents/skills/vllm-metax-branch-port/scripts/daily_config.py detect \
  --repo <meta-repo>
```

The helper detects the current branch, available remotes, GitHub remotes, `master`, and
all remote `mx/vX.Y.Z-dev` candidates. If a config exists, it validates it. If no config
exists, it returns only the unresolved questions and safe defaults.

Ask the user only for information that cannot be inferred safely:

- source and target refs when the candidates are ambiguous;
- all-source-only versus an explicit SHA subset;
- local-only versus approved remote publication, plus draft-pending approval only when remote publication is enabled;
- the accountable human scope owner and porting-developer identity;
- source-author policy for current and future commits;
- acceptable pending-validation categories and retry/time limits when overriding defaults.

After the answer, create a small JSON answers file and run:

```bash
python .agents/skills/vllm-metax-branch-port/scripts/daily_config.py create \
  --repo <meta-repo> --answers <answers.json>
```

Example answers file:

```json
{
  "source_ref": "mx/v0.32.0-dev",
  "target_ref": "master",
  "scope_mode": "all-source-only",
  "source_shas": [],
  "exclude_source_shas": [],
  "publication_mode": "local",
  "remote_publication_approved": false,
  "scope_approved_by": "<human-owner>",
  "porting_developer": "Name <email@example.com>",
  "source_author_policy": "current-branch-authors",
  "github_draft": false,
  "publish_with_pending_review": false,
  "draft_publication_with_pending_approved": false
}
```

This writes and validates `state/config.json` and `state/scope.json`. A later run
reuses the configuration without asking again.

## External agent state layout

All mutable branch-port state lives outside the repository. The default layout is:

```text
$METAX_AGENT_HOME/
└── <project>/
    ├── branch-port/
    │   ├── state/       # config, scope, matrix, daily state, leases and locks
    │   ├── runs/        # one flat directory per daily run
    │   ├── worktrees/   # run-scoped Git worktrees
    │   └── archive/     # optional audited backups; historical runs are discarded
    └── <other-agent>/   # independent state for other agent families
```

Environment overrides:

- `METAX_AGENT_HOME` changes the shared agent-state root.
- `METAX_BRANCH_PORT_HOME` is a legacy override for the complete branch-port directory.
- `METAX_BRANCH_PORT_POLICY_ROOT` points to the checkout containing `.agents` when the working repository does not carry the policy tree.

Do not create branch-port configuration, run artifacts, locks, or worktrees inside the source repository. Historical runs are disposable; the durable control state is kept in `state/`.

## Core checkpoints

A daily local run uses seven checks:

1. configuration and scope are valid;
2. source/target/master refs and source commits are frozen;
3. identity, branch, worktree and clean state are correct;
4. every source-only non-merge commit has a terminal disposition;
5. all-files pre-commit and necessary targeted checks pass;
6. the risk-tier review gate passes;
7. the final local commit and run state are recorded.

Remote authentication, PR-body validation, remote conflicts and publication receipts are
conditional checks enabled only by `publication.mode: auto`.

## Pre-approved configuration

Store the durable configuration outside tracked source, for example:

```text
$METAX_BRANCH_PORT_HOME/state/config.json
```

Start from [DAILY-CONFIG.example.json](../assets/DAILY-CONFIG.example.json) and
[DAILY-SCOPE.example.json](../assets/DAILY-SCOPE.example.json). For an explicit subset,
set scope `mode` to `explicit-shas` and list the selected source SHAs under
`source_shas`; list any explicitly excluded SHAs under `exclude_source_shas`.

The example files are the canonical schema. In addition to source, target and publication
approval, `state/config.json` records:

- porting-developer identity;
- approved kernel provider and environment fingerprint;
- source-author allowlist policy;
- risk-tiered review behavior, round limits and new-risk handling;
- accepted pending-validation categories;
- local versus remote publication mode;
- active-lease waiting policy;
- target-move and push-conflict retry policy;
- maximum unattended runtime.

The configuration approves only its named source/target scope, publication behavior and
how to handle pre-defined risk categories. It never authorizes merge, Gerrit submit, or a
new undisclosed design decision.

## Preflight before mutation

After fetching source, target and master refs, run:

```bash
python .agents/skills/vllm-metax-branch-port/scripts/daily_preflight.py \
  --repo <meta-repo> --config $METAX_BRANCH_PORT_HOME/state/config.json \
  --source-sha <source-sha> --target-sha <target-sha> --master-sha <master-sha> \
  --reviewer-capability available
```

The helper checks:

- configuration approvals and safety constraints;
- porting-developer identity;
- source-author policy and unknown authors;
- environment fingerprint and kernel provider;
- fresh independent-reviewer capability;
- remote publication/version/authentication checks only in `publication.mode: auto`;
- worktree-root writability and active edge leases.

`READY` continues. `WAIT` follows `active_lease_policy`. `BLOCKED` records artifacts and
performs no mutation.

## Run key and no-op behavior

Compute the durable run key with the deterministic daily-state helper:

```bash
python .agents/skills/vllm-metax-branch-port/scripts/daily_state.py \
  --repo <meta-repo> --config $METAX_BRANCH_PORT_HOME/state/config.json \
  --intent-file $METAX_BRANCH_PORT_HOME/state/scope.json \
  --source-sha <frozen-source-sha> --target-sha <frozen-target-sha> \
  --master-sha <frozen-master-sha> \
  --environment-fingerprint <approved-environment-fingerprint>
```

The helper hashes the source/target/master SHAs, configuration, intent file, policy,
agent/skill tree and environment fingerprint, then compares
`$METAX_BRANCH_PORT_HOME/state/daily-state.json`. If all inputs match a completed run and its
applicability/final review artifacts still validate, return `NOOP`. Do not recreate a
worktree merely because a new day started.

Record the final daily outcome with the same helper and `--record-status`, `--run-dir`,
`--target-head-sha` and repeated `--artifact` options.

Any source/target/policy/environment change creates a new run key and invalidates the old
audit and reviews.

## Worktree lifecycle

Never reuse fixed worktrees such as `vLLM-metax-pr-<name>`. Use the run-scoped manager:

```bash
python .agents/skills/vllm-metax-branch-port/scripts/worktree_manager.py prepare \
  --repo <meta-repo> --run-dir <run-dir> --edge-id <edge-id> \
  --source-sha <source-sha> --target-baseline-sha <target-sha> \
  --remote-pr-branch <stable-pr-branch-or-omit> \
  --worktree-root $METAX_BRANCH_PORT_HOME/worktrees --lease-hours 24
```

It creates:

```text
$METAX_BRANCH_PORT_HOME/worktrees/<edge-id>/<run-id>
```

The local branch is unique:

```text
port-run/<run-id>
```

The remote PR branch may remain stable and is updated later with `--force-with-lease`.
The manager records a lease in `$METAX_BRANCH_PORT_HOME/state/worktrees.json` and the run's
`WORKTREE.json`.

Heartbeat during long runs:

```bash
python .agents/skills/vllm-metax-branch-port/scripts/worktree_manager.py heartbeat \
  --repo <meta-repo> --run-dir <run-dir>
```

Validate the exact review checkout:

```bash
python .agents/skills/vllm-metax-branch-port/scripts/worktree_manager.py verify \
  --worktree-json <run-dir>/WORKTREE.json
```

Release after publication or a blocked/abandoned run:

```bash
python .agents/skills/vllm-metax-branch-port/scripts/worktree_manager.py release \
  --repo <meta-repo> --run-dir <run-dir> --status PUBLISHED
```

An active unexpired lease for the same edge blocks another mutation run. Concurrent daily
invocations must not overwrite the first run's worktree.

## Freshness rules

- Source/target/patch SHA changes invalidate semantic evidence immediately.
- Policy, daily configuration, agent or skill changes invalidate the run key.
- Environment fingerprint changes require environment re-verification.
- Worktree leases handle process ownership, not semantic validity.
- CI evidence is valid only for the exact head SHA and base SHA it ran against.

## Risk-tiered independent review

Record `risk_tier` in `RUN.json`:

- `LOW`: documentation, agent policy, build metadata or another non-runtime change. Use
  one fresh final reviewer after implementation. It combines target-first applicability,
  code review, falsification evidence and final SHA binding in `REVIEW.json`.
- `STANDARD`: ordinary behavior code. Use an applicability reviewer before implementation
  and a different fresh final reviewer afterward. The first gate decides whether the
  behavior is needed; the final gate checks implementation, tests and final SHA.
- `HIGH`: ABI, native kernels, distributed communication, cache lifetime, graph capture or
  quantization numerics. Use both gates and obtain the configured named human confirmation
  for material risks.

Do not repeat the complete target search in the final gate unless the target moved, the
implementation deviated from the approved intent, or a new source assumption was found.
Each gate must use the generic independent reviewer with
[reviewer-profile.md](reviewer-profile.md) in a fresh context; the implementer cannot
self-approve.

## Artifacts

Local runs keep four core files:

```text
RUN.json
AUDIT.md
REVIEW.json
VALIDATION.json
```

`REVIEW.json` is a superset validated by both the generic and adversarial review checkers.
`APPLICABILITY-REVIEW.json` exists only for `STANDARD` or `HIGH`. `PR_BODY.md` and
`PUBLISH.json` exist only for approved remote publication.

## Publication and cleanup

Publication is skipped entirely when `publication.mode` is `local`. For an approved remote
run, use `publish_branch_port.py` with the active lease and `REVIEW.json`:

```bash
--worktree-json <run-dir>/WORKTREE.json \
--adversarial-review <run-dir>/REVIEW.json
```

The publisher revalidates the artifact and requires the reviewed worktree to equal the
publishing checkout. It covers both Gerrit and GitHub.

After a successful or blocked run, release the lease. Schedule conservative cleanup:

```bash
python .agents/skills/vllm-metax-branch-port/scripts/worktree_manager.py cleanup \
  --repo <meta-repo> --older-than-days 14
```

The cleanup is dry-run unless `--apply` is supplied. It preserves active, dirty, unpushed,
open-review and non-terminal worktrees.
