# Daily branch-port automation

Use this reference for a scheduled, pre-approved end-to-end branch port. The human approves
a bounded scope and publication policy once in a durable configuration; each daily run is
then idempotent, run-scoped and blocked only by changed facts or unresolved risk.

## One daily entry point

Start one fresh Codex task from the repository root:

```text
Read .agents/agents/vllm-metax-branch-port-daily.md and execute the daily port using
<absolute-path-to-DAILY-CONFIG.json>. Do not request approval again for facts already
covered by that configuration. Do not merge or submit.
```

The daily coordinator automatically starts the independent reviewer in a fresh subagent
context. Manual reviewer startup is not part of the normal path.

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
- approval for automatic remote publication and draft publication with pending review;
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
  "remote_publication_approved": true,
  "scope_approved_by": "<human-owner>",
  "porting_developer": "Name <email@example.com>",
  "source_author_policy": "current-branch-authors",
  "github_draft": true,
  "publish_with_pending_review": true,
  "draft_publication_with_pending_approved": true
}
```

This writes and validates both `DAILY-CONFIG.json` and `DAILY-SCOPE.json`. A later run
reuses the configuration without asking again.

## External agent state layout

All mutable branch-port agent state lives outside the repository. The default layout is:

```text
/workspace/.metax-agent-state/
└── <project>/
    ├── branch-port/
    │   ├── control/      # DAILY-CONFIG, DAILY-SCOPE, DAILY-STATE, MATRIX, leases
    │   ├── runs/         # run artifacts and independent-review outputs
    │   ├── audits/       # durable audit snapshots
    │   ├── locks/        # cross-process locks
    │   └── worktrees/    # run-scoped Git worktrees
    └── <other-agent>/    # independent state for other agent families
```

Environment overrides:

- `METAX_AGENT_STATE_ROOT` changes the shared agent-state root.
- `METAX_AGENT_PROJECT` changes the project directory name.
- `METAX_BRANCH_PORT_HOME` overrides the complete branch-port directory.
- `METAX_BRANCH_PORT_POLICY_ROOT` points to the checkout containing `.agents` when the working repository does not carry the policy tree.

Do not create branch-port configuration, run artifacts, locks, or worktrees inside the source repository.

## Pre-approved configuration

Store the durable configuration outside tracked source, for example:

```text
$METAX_BRANCH_PORT_HOME/control/DAILY-CONFIG.json
```

Start from [DAILY-CONFIG.example.json](../assets/DAILY-CONFIG.example.json) and
[DAILY-SCOPE.example.json](../assets/DAILY-SCOPE.example.json). For an explicit subset,
set scope `mode` to `explicit-shas` and list the selected source SHAs under
`source_shas`; list any explicitly excluded SHAs under `exclude_source_shas`.

The example files are the canonical schema. In addition to source, target and publication
approval, `DAILY-CONFIG.json` records:

- porting-developer identity;
- approved kernel provider and environment fingerprint;
- source-author allowlist policy;
- review retry count and new-risk behavior;
- accepted pending-validation categories;
- draft publication with pending review;
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
  --repo <meta-repo> --config <DAILY-CONFIG.json> \
  --source-sha <source-sha> --target-sha <target-sha> --master-sha <master-sha> \
  --reviewer-capability available
```

The helper checks:

- configuration approvals and safety constraints;
- porting-developer identity;
- source-author policy and unknown authors;
- environment fingerprint and kernel provider;
- fresh independent-reviewer capability;
- publication/version inspection and remote authentication;
- worktree-root writability and active edge leases.

`READY` continues. `WAIT` follows `active_lease_policy`. `BLOCKED` records artifacts and
performs no mutation.

## Run key and no-op behavior

Compute the durable run key with the deterministic daily-state helper:

```bash
python .agents/skills/vllm-metax-branch-port/scripts/daily_state.py \
  --repo <meta-repo> --config <DAILY-CONFIG.json> \
  --intent-file <DAILY-SCOPE.json> \
  --source-sha <frozen-source-sha> --target-sha <frozen-target-sha> \
  --master-sha <frozen-master-sha> \
  --environment-fingerprint <approved-environment-fingerprint>
```

The helper hashes the source/target/master SHAs, configuration, intent file, policy,
agent/skill tree and environment fingerprint, then compares
`$METAX_BRANCH_PORT_HOME/control/DAILY-STATE.json`. If all inputs match a completed run and its
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
The manager records a lease in `$METAX_BRANCH_PORT_HOME/control/WORKTREES.json` and the run's
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

## Automatic adversarial review

The coordinator must:

1. run the applicability reviewer after the audit but before implementation;
2. create the run-scoped worktree only after the applicability review endorses the intent;
3. run the final reviewer after implementation, tests and latest-target rebase;
4. pass the exact worktree path, branch, target baseline SHA and head SHA to both gates;
5. validate both JSON artifacts with `check_adversarial_review.py`.

Use a fresh task/subagent for each reviewer gate. If the runtime cannot create an
independent context, daily automation stops with `BLOCKED`; it must not self-review.

## Publication and cleanup

Use `publish_branch_port.py` with both the active lease and final review:

```bash
--worktree-json <run-dir>/WORKTREE.json \
--adversarial-review <run-dir>/ADVERSARIAL-FINAL.json
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
