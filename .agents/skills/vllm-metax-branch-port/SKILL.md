---
name: vllm-metax-branch-port
description: Audit source-only commits from one older vLLM-MetaX development branch and carry relevant fixes or adaptations into one newer target branch under a maintained linear branch matrix, with confirmed Gerrit review or GitHub master PR publication. Use when asked whether commits on branch A still need to be ported to branch B; excludes full-version upgrades and direct Git branch merges.
---

# vLLM-MetaX branch port

Treat this as a semantic carry-forward workflow, not a Git branch merge or a blind
`cherry-pick`. Audit one user-specified source branch against one target branch, and
port only changes selected by the user after the audit. The default final target for
every branch-port invocation is GitHub `master`, even when an intermediate `mx/` edge
is requested first.

When acting as the coordinator, read the independent
[vllm-metax-branch-port agent](../../agents/vllm-metax-branch-port.md). It owns the
matrix, environment and validation sequence; this skill owns the audit and adaptation
rules.

## Invocation contract

The preferred invocation specifies only the starting ref and the final target:

```text
Use $vllm-metax-branch-port from mx/v0.30.0-dev to master; publish: yes/no.
```

The skill resolves every intermediate branch from the configured matrix and displays the
complete source-to-final-target chain for confirmation. The user does not need to list
`mx/v0.30.0-dev -> mx/v0.32.0-dev -> ... -> master` manually. An explicitly supplied
chain is an override and must be confirmed field-by-field.

Before auditing or mutating anything, show the resolved matrix as a table containing
every edge, logical source/target remotes, frozen SHAs where available, provider, target
public version, readiness, skipped `mx/` branches and the publication choice. Do not
silently omit an unavailable configured branch; report that edge as blocked.

```bash
python .agents/skills/vllm-metax-branch-port/scripts/publish_branch_port.py \
  plan-chain --repo . --source-ref mx/v0.30.0-dev --final-target master
```

By default branch existence comes from a live Gerrit remote query. If non-interactive
credentials are unavailable, the user must refresh the relevant remote-tracking refs
first, then rerun with `--use-tracking-refs`. Record that inventory source and never use
unrefreshed stale refs silently.

## Linear matrix policy

Use the configured matrix as a linear strong-guarantee chain. The current policy is:

```text
source mx/vX.Y.Z-dev
  -> every later existing mx/vX.Y.Z-dev, sorted by semantic version
  -> master
```

An `mx/` version exists only when its `mx/vX.Y.Z-dev` branch exists on the Gerrit remote.
Absent versions are skipped automatically; they do not create placeholder edges. Read
[linear-matrix.md](references/linear-matrix.md) before opening or closing an edge.

- One invocation owns the confirmed source-to-final-target chain while auditing one
  direct edge at a time. Audit only commits from that edge's direct source branch; do
  not reopen older source branches to search for omissions.
- An edge may be `CLOSED` only when every non-merge commit in the frozen source snapshot
  has a terminal disposition. `PORT_REQUIRED` and `NEEDS_REVIEW` are not terminal.
- A source commit that is version-specific, obsolete, out of scope or replaced by a
  target implementation produces no carry-forward commit. Its propagation stops at
  that edge and downstream edges do not reconsider the source commit.
- Downstream edges audit the target branch's actual commit snapshot, including native
  commits and adapted/replacement commits created by earlier edges.
- Do not open a downstream edge until its predecessor is `CLOSED` at the latest
  predecessor source SHA. Otherwise the linear strong guarantee is not established.
- A new source commit after closure reopens the predecessor edge. Process it before
  making another downstream completeness claim.
- Bind the final existing `mx/vX.Y.Z-dev -> master` edge to the master public version
  parsed from `setup.py`. Reopen and re-audit that edge when master changes its baseline.
- Record `final_target: master` on every run. Closing an intermediate edge is not chain
  completion; continue only after the predecessor changes are merged into the actual
  target branch so the downstream audit sees that branch's real snapshot.
- When a GitHub remote owns `master`, publish through a pull request. Its public version
  must be greater than or equal to every available `mx/vX.Y.Z-dev` branch version.
  Otherwise master is local-only.

## Remote publication

At the approval gate, ask explicitly whether the completed ports should be published to
the remote whenever the detected target provider is Gerrit or GitHub. This is a required
yes/no confirmation, not an inferred preference. If master has no GitHub remote, do not
offer a PR: present the detected local-only mode for confirmation. When publication is
possible and the answer is yes, read [publishing.md](references/publishing.md) and include
the confirmed plan in the audit approval.

- Before the audit, confirm the logical source and target remotes and their detected
  providers. Do not infer them silently.
- Only branches matching `mx/v<major>.<minor>.<patch>-dev` belong to Gerrit. A malformed
  `mx/` branch is blocked rather than treated as GitHub.
- `master` uses GitHub only when a GitHub remote owns the confirmed master baseline.
  Publish it only through a pull request; never push directly to `master`.
- If `master` is not owned by a GitHub remote, use local-only mode for master. Create the
  local commits and do not push or create a PR.
- Discover remotes and their provider roles from Git metadata. Refer to them only by
  logical remote name. Never write a repository URL, credentials or owner/repository URL
  into the skill, run state or reports.
- Push Gerrit changes to `refs/for/<target-branch>` after validation. Reuse each fresh
  `Change-Id`; do not submit automatically.
- Push a dedicated non-force branch to GitHub once the master version gate passes, then
  create or reuse an open pull request whose base is `master`. Do not merge it.
- Publication permission is limited to the confirmed provider and refs. Audit-only,
  local-only or a negative confirmation performs no remote mutation.

## Boundaries

- Require an explicit source ref and target ref. Do not infer a version from a name,
  installed package, or the current checkout.
- Before the formal audit, fetch only the named source and target branches from the
  selected remote and update their remote-tracking refs. Do not pull, checkout or move
  the user's local branches. Stop if the branches cannot be refreshed.
- Resolve and freeze full SHAs from the refreshed remote-tracking refs. Preserve the
  current checkout, index, dirty files and local commits.
- The audit is read-only with respect to source, local branches and the working tree.
  Do not prepare an environment, edit source, create a worktree, commit or publish until
  the user approves the audit and confirms the remote-publication choice.
- Ignore Gerrit merge wrapper commits. Audit their non-merge constituent commits.
- Do not claim a full framework upgrade. Use component-scoped evidence for a port.

## Audit first

1. Read the repository [AGENTS.md](../../../AGENTS.md) and the linked rules for
   any domain that the candidate commits touch.
2. Before evaluating commits, run `scripts/publish_branch_port.py inspect` to resolve
   and confirm the logical source/target remotes and provider. `mx/vX.Y.Z-dev` requires a
   unique Gerrit remote. `master` uses GitHub only when a GitHub remote owns its baseline;
   otherwise record local-only master publication.
3. Refresh the two requested branches before evaluating any commit. Use the repository
   remote, normally `origin`, and derive the branch names from refs such as
   `origin/mx/v0.30.0-dev`:

   ```bash
   git fetch --no-tags <remote> \
     "+refs/heads/<source-branch>:refs/remotes/<remote>/<source-branch>" \
     "+refs/heads/<target-branch>:refs/remotes/<remote>/<target-branch>"
   ```

   This updates only the two remote-tracking refs. The force prefix is limited to those
   refs so a remote branch rewrite is reflected accurately; it never force-updates a
   local branch. If the remote, authentication or either branch is unavailable, stop
   rather than auditing stale refs. Do not silently include unpushed local-only commits.
4. Immediately freeze the refreshed SHAs:

   ```bash
   git rev-parse --verify "refs/remotes/<remote>/<source-branch>^{commit}"
   git rev-parse --verify "refs/remotes/<remote>/<target-branch>^{commit}"
   ```

   Record the UTC fetch time and these full SHAs in the audit. Do not re-resolve the
   branch names while the audit is in progress.
5. Create a read-only inventory from the frozen SHAs:

   ```bash
   python .agents/skills/vllm-metax-branch-port/scripts/inventory_commits.py \
     --repo . \
     --source <source-sha> \
     --target <target-sha> \
     --output .metax-branch-port/runs/<source-slug>__to__<target-slug>/inventory.json
   ```

6. Review every source-only non-merge commit without pre-filtering. The audit report must
   contain one row for every commit in the inventory, including commits believed to be
   already adapted, superseded or no longer needed. Do not omit a commit because its
   patch ID matches, its subject looks familiar, or an initial inspection suggests no
   action. Read the full diff, original author, issue references, changed call paths,
   tests and relevant upstream history.
7. Classify each commit using the evidence rules in
   [audit-rubric.md](references/audit-rubric.md). Never use a merge conflict alone as
   proof that a change is version-specific or obsolete. For every commit, state the
   evidence found on the target and whether migration or target-specific adaptation is
   required. For every `PORT_REQUIRED` commit, complete the detailed logic audit in
   [commit-logic-audit.md](references/commit-logic-audit.md), including a simple
   user-facing example that explains the source behavior and expected adapted behavior.
   The example is explanatory, not a required numeric test or reproduced result. If the
   logic cannot be explained clearly, use `NEEDS_REVIEW` instead of `PORT_REQUIRED`.
8. Write `.metax-branch-port/runs/<source-slug>__to__<target-slug>/AUDIT.md` from the
   template in the rubric. Include the frozen SHAs, every source-only commit, merge
   wrappers ignored, target evidence, original authors, proposed conventional type
   and scope, required validation and dependencies. Validate that the number of commit
   decision rows equals `inventory.commit_count`; a missing row makes the audit
   incomplete. Also record the matrix edge ID, predecessor status and resulting edge
   status. Append one `Commit logic audit` section for every commit currently marked
   `PORT_REQUIRED`.
   Include the confirmed step 2 redacted provider, logical remote, Gerrit review ref or
   GitHub PR plan, local-only master decision when applicable, final master version gate,
   and the required yes/no publication confirmation in the audit.
9. Run the completeness checker against the inventory and audit:

   ```bash
   python .agents/skills/vllm-metax-branch-port/scripts/check_audit_completeness.py \
     --inventory .metax-branch-port/runs/<source-slug>__to__<target-slug>/inventory.json \
     --report .metax-branch-port/runs/<source-slug>__to__<target-slug>/AUDIT.md \
     --port-sha <source-sha-for-each-port-required-commit>
   ```

   Repeat `--port-sha` for every `PORT_REQUIRED` commit. The helper must report every
   inventory commit as present and confirm each port candidate has a logic-audit section
   containing an explanation example. Do not present an incomplete audit for approval.

By default, bug fixes, performance fixes, model adaptations, required build/dependency
compatibility and security fixes are port candidates. Ordinary features, documentation,
formatting and test-only changes are reported but are not automatic port candidates.
`PORT_REQUIRED` still requires user selection; uncertainty is `NEEDS_REVIEW`.

## Approval gate

Present the complete audit, detailed port logic explanations and proposed selected
commits before mutation. The user must approve:

- the exact source commits to port and their order;
- the target branch and full target SHA;
- the target vLLM ref, full commit and public version;
- the target MetaX public version and confirmation that it equals the vLLM version;
- whether to publish to the remote, the detected provider and logical remote name, and
  the Gerrit target ref or GitHub PR base/head mapping;
- confirmation of the logical source and target remotes, or explicit confirmation that
  `master` has no GitHub remote and publication is local-only;
- confirmation that `final_target` is GitHub `master` and that its public version is at
  least the latest available `mx/vX.Y.Z-dev` public version;
- the original commit authors' exact names and email addresses;
- the porting developer's exact name and email;
- confirmation with the source author that this developer may carry the selected
  changes forward.

Do not infer the porting developer from global Git configuration. On `mx/` targets,
preserve each source commit's original author exactly and use the confirmed porting
developer as committer and `Signed-off-by`. On `master`, use the confirmed porting
developer as both author and committer. Verify both identities after every commit. If a
selected commit has several authors or the confirmation cannot be established, stop
before changes and ask.

After approval, read [porting-and-commit.md](references/porting-and-commit.md) before
starting the mutation phase.

## Environment and adaptation

1. Create an isolated worktree from the re-fetched, frozen target SHA before preparing
   the environment. Do not reuse or modify the user's active checkout.
2. Read and apply
   [independent-environment.md](references/independent-environment.md). Obtain one
   complete branch-port environment approval, then install and verify the target
   environment before adaptation. Include Git identity, Gerrit `commit-msg` hook,
   upstream vLLM mapping, the MetaX/vLLM public-version equality gate, and the confirmed
   publication provider/auth prerequisites.
3. Install target MetaX first, using the cleaned public build requirements, then install
   matching upstream vLLM. Both must succeed and verify before adaptation.
4. Register the selected work in the independent branch-port matrix and run state. Do not
   create a common-sense or `vllm-metax-upgrade` full-upgrade run.
5. For work spanning several domains, continue as the branch-port coordinator while
   retaining the selected-commit scope.
6. Before making an adaptation patch, read and apply
   [common-sense](../common-sense/SKILL.md) and every reference it requires for source
   ownership, upstream-first structure, dependency closure, validation and recovery.
   Use common-sense rules without creating a full-upgrade run.
7. Route adaptation to the owning specialist skill:
   - model implementation: `model-adaptation` or `model-upgrade`;
   - attention/cache: `attention-upgrade`;
   - registry/quantization/CustomOps: `registry-upgrade`;
   - csrc/native/build: `csrc-upgrade`;
   - `vllm_metax/patch/`: `patch-upgrade`;
   - shared source rules and durable state: `common-sense`.
8. Reproduce the original intent against target APIs and contracts. Port semantics,
   not source text. Do not add unrelated cleanup or optimization.

## Commits and validation

- Produce one target commit for each selected source commit. Do not batch multiple
  source commits into one target commit.
- Reconstruct the intent from the actual change and evidence, then write a new
  Conventional Commit message. Do not copy an informal source subject verbatim.
- Do not preserve the source `Change-Id`.
- For an `mx/vX.Y.Z-dev` target, preserve the original author and add exactly enough
  provenance with `Source: <source-ref>@<full-sha>`.
- For `master`, use the confirmed porting developer as both author and committer, and
  omit the `Source:` trailer. Preserve source-to-target traceability in `MATRIX.json`,
  `VALIDATION.md` and the pull request body instead.
- Let the Gerrit hook generate one new `Change-Id` for every target commit. Verify each
  commit separately.
- On `mx/` targets, preserve both the original author name and email. Use the approved
  porting developer as committer and signer-off. Never fabricate either identity.
- Use conservative, tiered validation. Run static checks, targeted tests and relevant
  component tests by default. Run e2e or benchmarks only when the change affects a path
  that targeted checks cannot establish, or when a performance claim requires it.
- If a selected commit depends on another source commit, preserve and validate the
  dependency order. Do not silently produce a partial port.

At completion, report source-to-target SHA mapping, new `Change-Id` values, exact
original author names and emails, target-specific adaptations, validation evidence,
skipped commits, publication result and unresolved risks. Update the matrix only after
every source commit has its final disposition and the edge's validation is complete. Do
not mark an edge `CLOSED` while any selected port, review item or stale baseline remains.
Remote publication occurs only under the yes answer captured at approval. `master` PR
creation completes the master publication phase; it does not authorize merge or submit.
