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
  validation evidence and final edge status under `.metax-branch-port/`.
- Do not initialize or advance the `vllm-metax-upgrade` coordinator.
- Do not require a full framework upgrade or a full csrc upgrade. Review selected csrc
  commits with `csrc-upgrade`, but keep the claim limited to the approved port.
- Before any adaptation patch, fully read and apply `common-sense` and its required
  references for upstream-first source ownership, dependency closure, validation and
  recovery. This does not create or authorize a full-upgrade run.
- Reuse shared domain skills for changed contracts.
- Preserve user worktrees, identities and uncommitted changes. Do not push unless the
  user explicitly requests it after reviewing local commits.

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

Do not infer the upstream mapping from a branch name. If the versions differ, mark the
edge `BLOCKED`; do not silently edit `setup.py`, `requirements/common.txt` or dependency
files to force installation. A version synchronization is a separately scoped change
that requires audit and approval.

## Workflow

1. Read the current matrix and verify that the predecessor edge is closed through its
   latest source SHA.
2. Refresh only the named source and target MetaX branches, freeze full SHAs, run the
   read-only commit inventory and produce an exhaustive audit. Every inventory
   non-merge commit, including already-adapted or likely-unneeded commits, must have its
   own row with target evidence and a migration/adaptation conclusion.
3. For every `PORT_REQUIRED` commit, produce a complete logic audit with a simple
   user-facing explanation example as specified by the branch-port skill. The example
   explains the behavior; it is not a mandatory numerical test or reproduced result. Do
   not infer a port decision from a diff summary alone. If the logic or explanation is
   incomplete, mark the commit `NEEDS_REVIEW`.
4. Present the complete audit, per-commit logic explanations, selected commits, original
   authors, porting developer identity and source-author confirmation request. Wait for
   approval.
5. Re-fetch both MetaX branches and verify their frozen SHAs. If either moved, invalidate
   the affected audit and request approval for any changed selection or baseline.
6. Create an isolated target-branch worktree before preparing the environment.
7. Prepare one complete branch-port environment plan for user approval. Follow
   [independent-environment.md](../skills/vllm-metax-branch-port/references/independent-environment.md).
8. Install target `vllm_metax` first, then the matching target vLLM. Both installations
   must succeed and verify before adaptation.
9. Before patching, read and apply `common-sense`. Then port approved commits in
   dependency order. Read the owning model, attention, registry, csrc or patch skill for
   each changed contract. Keep one target commit per selected source commit.
10. Run conservative validation. Record current evidence, exact source/target SHAs,
   authors, new `Change-Id` values and environment fingerprint.
11. Close the matrix edge only when every source non-merge commit has a terminal
    disposition, all selected ports and validations are current, and both baselines and
    version bindings remain unchanged.

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
