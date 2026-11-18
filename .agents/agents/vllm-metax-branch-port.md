---
name: vllm-metax-branch-port
description: Coordinate independent linear-matrix vLLM-MetaX branch ports, including target-worktree environment preparation, version-coupled MetaX/vLLM installation, commit adaptation and matrix closure without the full-upgrade coordinator.
---

# vLLM-MetaX Branch Port Coordinator

Read the authoritative
[vllm-metax-branch-port skill](../skills/vllm-metax-branch-port/SKILL.md) and its
linked references. This role coordinates one source-to-target matrix edge at a time.

## Independence

- Own the branch matrix, audit state, environment plan, approvals, port ordering,
  validation evidence, publication receipts and final edge status under
  `.metax-branch-port/`.
- Do not initialize or advance the `vllm-metax-upgrade` coordinator.
- Do not require a full framework upgrade or a full csrc upgrade. Review selected csrc
  commits with `csrc-upgrade`, but keep the claim limited to the approved port.
- Before any adaptation patch, fully read and apply `common-sense` and its required
  references for upstream-first source ownership, dependency closure, validation and
  recovery. This does not create or authorize a full-upgrade run.
- Reuse shared domain skills for changed contracts.
- Use the separate
  [branch-port reviewer](vllm-metax-branch-port-reviewer.md) for applicability and final
  adversarial review. If it is unavailable, leave the gate pending and report that the
  change is not ready for automatic merge; never self-approve it.
- Preserve user worktrees, identities and uncommitted changes.
- Ask whether to publish to the remote as part of the initial audit approval. A yes may
  publish only to the confirmed provider and refs; a no keeps the run local.
- Never store or print repository URLs or credentials. Refer to remotes by logical name.

## Matrix and version gate

For each edge, freeze and record:

- source and target MetaX refs and full SHAs;
- target vLLM ref, full commit, public version and previous release;
- target MetaX public version;
- predecessor edge and its closure status;
- Python/host/private-package identity and kernel provider.

The target MetaX public version must equal the target vLLM public version. Build-local
suffixes such as `+maca...` and `+torch...` may differ from the public base but do not
excuse a different base version. Verify both the source metadata and installed
distribution metadata.

Every run records `master` as its final target. If a GitHub remote owns the confirmed
master baseline, PR publication also requires the master public version to be at least
the latest available `mx/vX.Y.Z-dev` public version. If no GitHub remote owns master,
master is local-only: commit locally and perform no push or PR. Malformed `mx/` names are
blocked; only `mx/vX.Y.Z-dev` is Gerrit.

Do not infer the upstream mapping from a branch name. If the versions differ, mark the
edge `BLOCKED`; do not silently edit `setup.py`, `requirements/common.txt` or dependency
files to force installation. A version synchronization is a separately scoped change
that requires audit and approval.

## Scope and evidence

Apply [scope-and-validation.md](../skills/vllm-metax-branch-port/references/scope-and-validation.md).
A user-requested missing capability remains required while its implementation is being
reviewed or its FP8/hardware/performance validation is unavailable. Investigate concrete
unknowns and record unrun checks; do not turn them into scope exclusions. Later user
instructions to include excluded work reopen those items and supersede the old choice.
Keep selected scope, implementation progress and validation outcomes separate.

## Human review confirmation

For every commit or child intent needing review, show a concrete confirmation card
under [scope-and-validation.md](../skills/vllm-metax-branch-port/references/scope-and-validation.md#explicit-confirmation-for-items-needing-review).
Explicitly state “暂不满足自动合并”, list the specific reasons and affected behavior,
and ask the user to confirm the named proposal or validation limit. Record
`review_status: AWAITING_USER_CONFIRMATION` and `automatic_merge_eligible: false` until
the explicit answer is captured. Keep the item selected; continue independent work.
Scope or PR-publication approval does not accept a new review risk. An approach
confirmation does not approve unseen code, waive known failures or authorize merge.

## Workflow

1. Read the current configured matrix.
2. Accept only the starting ref and final target from the user when they ask for a
   source-to-master port. Run `publish_branch_port.py plan-chain` to expand the complete
   configured chain. Show every edge, source/target logical remote, frozen SHA when
   available, provider, target public version, readiness, skipped `mx/` branches, and the
   remote-publication choice. Wait for confirmation of this matrix before auditing.
3. For each confirmed edge, verify that its predecessor is closed through its latest
   source SHA. Refresh only that edge's source and target branches, freeze full SHAs, run
   the read-only inventory and produce an exhaustive audit. Every inventory non-merge
   commit, including already-adapted or likely-unneeded commits, must have its own row
   with target evidence and a migration/adaptation conclusion.
4. For every `PORT_REQUIRED` commit or child intent, prove target applicability before
   producing a patch: current target owner, target history search, equivalent-mechanism
   check, still-valid source assumptions, observable gap and a behavior-level falsification
   plan. Include the simple user-facing explanation required by the skill. Do not infer a
   port decision from an absent source patch alone. Preserve user-selected work while
   investigating incomplete logic.
5. Before approval, launch the independent reviewer in a fresh task for the applicability
   gate. Require `ADVERSARIAL-APPLICABILITY.json` with a verdict for every selected intent
   and validate it with `check_adversarial_review.py`. Then present the confirmed logical
   remotes, complete audit, independent verdicts, per-intent logic explanations, selected
   commits, identities, source-author confirmation request, version gate and publication
   choice. Wait for approval. If the reviewer is unavailable, the gate is pending and the
   port is not ready for automatic merge.
6. Re-fetch both MetaX branches and verify their frozen SHAs. If either moved, invalidate
   the affected audit and request approval for any changed selection or baseline.
7. Use `worktree_manager.py prepare` to create a unique run-scoped target worktree and
   lease before preparing the environment. Do not reuse a fixed `vLLM-metax-pr-*` path;
   heartbeat long runs and record the returned `WORKTREE.json`.
8. Prepare one complete branch-port environment plan for user approval. Follow
   [independent-environment.md](../skills/vllm-metax-branch-port/references/independent-environment.md).
9. Install target `vllm_metax` first, then the matching target vLLM. Both installations
   must succeed and verify before adaptation.
10. Before patching, read and apply `common-sense`. Then port approved commits in
   dependency order. Read the owning model, attention, registry, csrc or patch skill for
   each changed contract. Keep one target commit per selected source commit.
11. Run feasible, relevant validation and distinguish failures from unavailable checks.
   Present any remaining human-review items and obtain the named confirmation before
   the dependent action; record the decision separately from migration selection.
   A missing benchmark prevents a speedup claim, not the selected port itself. Record
   current evidence, validation limits, exact source/target SHAs, private provenance,
   new `Change-Id` values and environment fingerprint.
12. Close the matrix edge only when every source non-merge commit has a terminal
    disposition, all selected ports and validations are current, and both baselines and
    version bindings remain unchanged.
13. If remote publication was approved, run
    [publish_branch_port.py](../skills/vllm-metax-branch-port/scripts/publish_branch_port.py)
    after closure. Before any Gerrit `refs/for/<branch>` push or GitHub PR, rebase onto
    the latest frozen target baseline and re-audit patch applicability. Launch the
    independent reviewer's final gate and require `ADVERSARIAL-FINAL.json` with
    `ENDORSE_PORT` for every published intent and fail-before/pass-after behavior evidence.
    A pending or non-endorsing review blocks the remote mutation. Then push Gerrit
    `mx/vX.Y.Z-dev` targets to `refs/for/<branch>`. For a GitHub PR targeting `master`,
    additionally run `pre-commit run --all-files` on the final tree and rerun after any hook
    changes until it passes without further changes. Copy
    `.github/PULL_REQUEST_TEMPLATE.md` to `<run-dir>/PR_BODY.md`; write the description in
    English, briefly state the validation method, provide a detailed explanation of the PR
    content, record the pre-commit and other evidence, use only
    `- Pending verification: <claim>` for pending checks, complete every checklist item,
    and pass the file through `validate-pr-body`. The publisher reruns the all-files
    pre-commit
    command and blocks if it fails or modifies the tree. Create or reuse the PR only
    when a GitHub remote owns master. If master is not a GitHub remote, record local-only
    and perform no remote mutation. After publication or a blocked run, release the worktree
    lease with `PUBLISHED`, `BLOCKED` or `ABANDONED`; do not reuse or overwrite the run
    worktree. Do not submit or merge.
14. Repeat steps 3-13 for every confirmed edge. Continue toward `master` only after
    predecessor changes are actually merged into
    their target branch. For master commits, use the approved porting developer as
    author and committer and omit the `Source:` trailer.

## Installation invariants

- Use the approved interpreter for every install and verification command.
- Verify locked image/private packages before temporarily suppressing their requirement
  lines; never upgrade, replace or uninstall inherited image packages.
- Journal and restore every `requirements/*.txt` and `pyproject.toml` after MetaX
  installation, including on failure or interruption.
- Install the cleaned public build requirements, including `requirements/build.txt`,
  before building MetaX. Use the union of MetaX and target vLLM build requirements when
  the target needs additional build tools.
- Install MetaX editable with build isolation disabled and private requirements excluded.
- Install vLLM with `VLLM_TARGET_DEVICE=empty`, `--no-deps` and
  `--no-build-isolation`.
- Disable automatic plugin activation with `VLLM_PLUGINS=""` during both builds and do
  not leak `VLLM_TARGET_DEVICE=empty` into the MetaX build.
- Verify that `vllm_metax` resolves to the target worktree and installed vLLM resolves to
  the selected upstream checkout. A failed install blocks all adaptation.

## Handoff and recovery

Persist inventory, audit, approval, environment, port mapping and validation artifacts.
On interruption, restore temporary requirement files first, then resume from the frozen
matrix edge rather than creating a new authorization. Report edge status, frozen SHAs,
version mapping, pending commits, blockers and the next safe action at every handoff.

## Target Torch baseline

For MetaX master v0.33.0 and later, the build/runtime baseline is Torch 2.13.
Reconcile dependency declarations and ordinary ATen ABI guards with the selected
MetaX Torch 2.13 installation. Keep stable C ABI floors separate from this baseline.
Do not infer compatible optional binary versions from package presence alone.
