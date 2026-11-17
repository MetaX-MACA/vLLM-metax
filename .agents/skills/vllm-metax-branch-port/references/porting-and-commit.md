# Port approval, adaptation and commits

Read this reference only after the user approves the read-only branch audit.

## Identity gate

Before changing the target worktree, confirm all of the following in one user-facing
approval:

- every source commit selected for porting and its original author's exact name and
  email address;
- the porting developer's exact Git author name and email;
- that the source author has confirmed this developer may carry the change forward;
- any target-specific adaption or feature-scope decision recorded by the audit.
- whether to publish to the remote and, when yes, the provider, logical remote name,
  Gerrit review ref or GitHub PR base/head mapping;
- that `master` is the final target and, when GitHub owns it, its public version is at
  least the latest available `mx/vX.Y.Z-dev` public version; otherwise publication is
  local-only.

Preserve the source author's name and email exactly with:

```text
--author="<original-name> <original-email>"
```

Set the confirmed developer as the committer and `Signed-off-by`, for example:

```bash
git -c user.name="<porting-name>" -c user.email="<porting-email>" \
  commit --author="<original-name> <original-email>" -s
```

Do not use an inferred, inherited or empty identity. Different source authors must
remain distinct in the source-to-target mapping. After each commit, verify both
identities:

```bash
git show -s --format='author=%an <%ae>%ncommitter=%cn <%ce>' <target-sha>
```

For an `mx/` target, author name and email must equal the source commit metadata. For
`master`, author and committer must both equal the approved porting developer. In every
case the committer and `Signed-off-by` use the porting developer's exact identity. Stop
and correct the commit if any field differs.

This source-author preservation applies to `mx/vX.Y.Z-dev` targets. For a commit created
on `master`, use the confirmed porting developer as both author and committer and omit
the `Source:` trailer. The matrix and pull request body retain provenance.

## Environment and worktree

Use the independent branch-port environment workflow in
[independent-environment.md](independent-environment.md). Its single preparation plan
must include the target MetaX worktree, explicit vLLM ref/commit/version, Python
environment, installed frameworks, private package locks, kernel provider, chip scope,
Git identity, Gerrit commit-message hook, and the authentication/tool prerequisites for
the confirmed publication provider. Master publication has no GitHub prerequisite when
the confirmed mode is local-only. Audit-only work does not need this gate.

MetaX and vLLM public versions must match. A mismatch blocks installation and adaptation;
do not auto-edit version or common-requirement files as part of environment preparation.

Immediately before creating the worktree, fetch the same source and target branch refs
again and compare their full SHAs with the audit's frozen SHAs. If either branch moved,
do not create the port worktree from the stale audit. Re-run the commit inventory,
review the new commits or rewritten history, update the audit, and ask for approval
again if the selected source commits or target baseline changed.

Also confirm that the predecessor matrix edge is `CLOSED` through the latest
predecessor source SHA. Do not use a complete target snapshot as evidence for the
linear guarantee if the edge that was supposed to populate it still has pending or
unreviewed source commits.

Use an isolated worktree based on the approved full target SHA. Name the local work
branch explicitly and record it in the run state. Keep the source checkout available
for evidence, but do not alter it.

The Gerrit `commit-msg` hook must create a new `Change-Id`. If the hook is absent or the
repository is configured with a pre-commit-only hook, resolve it under the approved
environment plan. Do not write a fabricated `Change-Id`, and do not reuse one from the
source.

## Domain ownership

Use the independent branch-port matrix and run state for durable source mapping,
ownership and acceptance. Do not create a full-upgrade coordinator run. Read the exact
specialist skill for each changed contract:

- `model-adaptation`: new model code or independent model implementation dependencies;
- `model-upgrade`: existing model compatibility across revisions;
- `attention-upgrade`: attention backend, MLA, sparse indexer, cache shape and wrapper;
- `registry-upgrade`: registration, quantization, CustomOps and kernel dispatch;
- `csrc-upgrade`: native source, headers, bindings and build contracts;
- `patch-upgrade`: files and activation paths under `vllm_metax/patch/`.

Before any adaptation patch, fully read and apply `common-sense`. Follow its
upstream-first source structure, complete file ownership, dependency closure, validation
and recovery rules, together with any linked references needed for the affected domain.
This use does not authorize or require a full-upgrade run.

Port only approved commits. A shared file changed by several commits needs coordinated
ownership and ordered acceptance; it does not justify combining their target commits.

## Target commit message

Derive the target subject from what the code actually fixes or enables, not from an
informal source subject. Use these types:

- `fix`: correctness, crash, hang, compatibility defect;
- `perf`: measured or structurally justified performance change;
- `feat`: required model or capability adaptation;
- `build`: dependency, packaging or build-system change;
- `refactor`: behavior-preserving structural change needed for the port;
- `test`, `docs`, `style`, `chore`, `ci` only when explicitly approved.

Use a subsystem scope where it materially helps, for example:

```text
fix(attention): handle sparse MLA padding
perf(csrc): remove redundant decode synchronization
feat(model): support GLM5 indexer on v0.32
build(deps): align mixl with the target image
```

The body should explain the target symptom or need, the adaptation made and any
important contract difference. For an `mx/vX.Y.Z-dev` target, preserve traceability with
a trailer:

```text
Source: mx/v0.30.0-dev@<full-source-sha>
```

Do not copy the original `Change-Id`. Each target commit receives a fresh one from the
Gerrit hook:

```text
<type>(<scope>): <target intent>

<why the target still needs this and what changed during adaptation>

Source: <source-ref>@<full-source-sha>
Signed-off-by: <confirmed porting developer>
Change-Id: <new value generated by Gerrit hook>
```

For `master`, author and committer are both the confirmed porting developer, and the
commit has no `Source:` trailer:

```text
<type>(<scope>): <target intent>

<why the target still needs this and what changed during adaptation>

Signed-off-by: <confirmed porting developer>
```

Record source provenance in `MATRIX.json`, `VALIDATION.md` and the PR body.

Use one commit per selected source commit. Commit dependencies in the approved order.
Do not squash related commits or split one source commit without explicit approval.

## Conservative validation

Choose the minimum evidence that actually proves the target behavior:

1. Static: review the target-vs-source semantic mapping, imports, API contracts and
   changed-file lint. Run focused unit or component tests covering the changed behavior.
2. Runtime: when dispatch, cache layout, quantization, operators, model loading or
   distributed behavior is affected, exercise the actual selected component and record
   the configuration and observed backend.
3. E2E/benchmark: use only when the earlier tiers cannot establish correctness, or when
   the commit's value depends on a performance or full-model result. Record the exact
   model, hardware, configuration and compare against the frozen target baseline.

Run `pre-commit` on the changed files after every commit or at least before declaring
the selected run complete. Re-run any evidence invalidated by later edits. A green exit
code without asserting the expected behavior is not sufficient.

When publication was approved, validate provider authentication and idempotency before
the remote mutation. For Gerrit, verify every new commit has its own `Change-Id` and push
only to `refs/for/<mx-branch>`. For GitHub, verify the master version gate, push a
dedicated non-force branch, and create or reuse an open PR targeting `master`.

## Completion

Report:

- source ref/SHA to target commit SHA mapping;
- UTC fetch time and confirmation that both branches still matched their frozen SHAs
  immediately before adaptation;
- original author and porting developer exact names and email addresses for each commit;
- each new `Change-Id`;
- adaptations and dependency ordering;
- validation commands, configurations and outcomes;
- skipped or blocked commits with their audit evidence;
- remaining risks or unverified hardware/model combinations;
- publication status, logical remote name and either the Gerrit review ref or GitHub PR
  number, with no repository URL.

Close the matrix edge only after all source commits have a terminal disposition, every
approved port has a verified target SHA and `Change-Id`, both baselines remain current,
and the required validation is complete. If new source commits arrived after the frozen
snapshot, reopen the edge and audit the increment before starting a downstream edge.

Make no claim beyond the selected component scope. Remote publication occurs only under
the yes answer captured at audit approval. Never submit a Gerrit change or merge a
GitHub PR automatically.
