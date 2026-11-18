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

## Daily workflow

1. Run `daily_config.py detect --repo <meta-repo>`. If no config exists, present the
   detected branch candidates and only the unresolved questions to the user, then write an
   answers file and call `daily_config.py create`. Verify the resulting configuration,
   including scope approval, publication approval and `merge_authorized: false`, before
   continuing.
2. Fetch only the configured source, target and master refs. Freeze full SHAs.
3. Run `daily_preflight.py` with `--reviewer-capability available`. If it returns `WAIT`,
   follow `worktree.active_lease_policy` and wait up to the configured limit, then rerun
   preflight. If it returns `BLOCKED`, record the blockers and stop before mutation.
4. Run `daily_state.py` to compute the run key from source/target/master SHAs, selected
   intents, configuration hash, policy/skill tree and environment fingerprint.
5. Return `NOOP` when `daily_state.py` reports the same completed run key and the recorded
   artifacts still validate.
6. Run the read-only audit and target-first applicability analysis. Do not create a
   worktree or mutate code yet.
7. Start `vllm-metax-branch-port-reviewer` in a fresh subagent/task for the applicability
   gate. Wait for `ADVERSARIAL-APPLICABILITY.json` and validate it with
   `check_adversarial_review.py`. If no independent context is available, stop `BLOCKED`.
8. Only after endorsement, create a run directory and unique worktree through
   `worktree_manager.py prepare`. Record the lease path and run key.
9. Prepare the approved environment and implement the selected intents in the run-scoped
   worktree. Heartbeat the lease during long work.
10. Rebase to the latest frozen target baseline. Repeat target-first applicability and the
   behavior-level falsification check. Heartbeat the worktree with the final local head SHA
   so the lease record matches the commit that will be reviewed and published.
11. Start the same reviewer role in another fresh subagent/task for the final gate. Require
    `ADVERSARIAL-FINAL.json` with `ENDORSE_PORT` for every published intent and validate
    the exact worktree, branch, target baseline and head SHA.
12. Prepare the English PR body when the provider is GitHub. Run all-files pre-commit when
    required. Publish through `publish_branch_port.py` with
    `--worktree-json <run-dir>/WORKTREE.json` and
    `--adversarial-review <run-dir>/ADVERSARIAL-FINAL.json` for both Gerrit and GitHub.
13. Release the worktree lease as `PUBLISHED`, `BLOCKED` or `ABANDONED`, then record the
    outcome with `daily_state.py --record-status ...` and its run/artifact paths. Report the
    no-op/blocked result, source/target/head SHAs, review verdicts, publication receipt and
    cleanup candidates.

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
