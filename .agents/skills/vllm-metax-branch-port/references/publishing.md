# Remote publication

Read this reference when the user answers **yes** to remote publication at the audit
approval gate.

## Authorization and scope

The approval must name:

- `publish_requested: true`;
- the detected provider and logical remote name;
- the Gerrit review ref or GitHub PR base/head branches;
- the final GitHub `master` version gate;
- the master commit identity policy.

Publication is limited to those refs. A negative answer, an audit-only request or an
unconfirmed provider performs no remote mutation.

Never write a repository URL, credentials, token, owner/repository URL or PR URL into the
skill, `MATRIX.json`, `VALIDATION.md` or logs. Use remote names such as `origin` or
`upstream` only after discovering their roles from Git metadata. Redact any URL emitted
by Git or `gh`.

## Branch and provider contract

Only the branch contract selects the provider:

- `mx/v<major>.<minor>.<patch>-dev` -> Gerrit.
- `master` with a confirmed GitHub remote -> GitHub PR.
- `master` without a GitHub remote -> local-only commits, no push or PR.
- Any other branch -> local-only.
- A name beginning with `mx/` that does not match the strict development-branch pattern
  is blocked; do not guess GitHub or a malformed Gerrit ref.

Before the audit approval, run:

```bash
python .agents/skills/vllm-metax-branch-port/scripts/publish_branch_port.py inspect \
  --repo . --source-ref <source-ref> --target-ref <target-ref>
```

The helper prints redacted JSON containing the provider, logical remote, review/PR plan,
latest `mx/` public version, master public version, version-gate result and an explicit
local-only reason when master is not a GitHub remote.

## Master version gate

For a `master` target, parse its public branch version from `setup.py` in the confirmed
GitHub or local master tree. Parse every remote `mx/vX.Y.Z-dev` branch version and select
the highest semantic version. PR publication is allowed only when:

```text
master_public_version >= latest_mx_dev_public_version
```

Build-local suffixes are not part of this comparison. A malformed or missing version or
an absent master baseline blocks the run. An absent GitHub remote changes the result to
local-only instead of blocking publication.

## Gerrit publication

After all local validation passes and every selected commit has a fresh, unique
`Change-Id`:

1. Resolve the logical Gerrit remote from the target branch's remote-tracking ref.
2. Confirm the target still matches the final frozen SHA used by the run.
3. Check the supplied Gerrit hook and identity gates.
4. Push without force:

   ```bash
   git push --porcelain <logical-remote> HEAD:refs/for/<mx-target-branch>
   ```

5. Record `publication_status: pushed`, the logical remote, target branch and review ref.
   Do not record the remote URL.
6. Do not submit the change. Human review and submission remain required.

The helper command is:

```bash
python .agents/skills/vllm-metax-branch-port/scripts/publish_branch_port.py publish \
  --repo . --run-dir <run-dir> --target-ref <mx-target-branch>
```

## GitHub master publication

GitHub publication uses a dedicated non-force branch and a PR whose base is `master`.
Never push to `master` directly and never merge the PR automatically.

Use this section only when the confirmed logical GitHub remote owns master. Otherwise
follow the local-only rule below.

1. Verify that `master` passes the version gate.
2. Verify the GitHub remote and a working authenticated `gh` CLI before pushing.
3. Use the approved porting developer as both author and committer for every master
   commit. Omit `Source:` trailers.
4. Push a dedicated branch without force:

   ```bash
   git push --porcelain <github-remote> HEAD:refs/heads/<pr-branch>
   ```

5. Search for an open PR with the same head branch and base `master`.
6. Reuse the open PR when present. Otherwise create one with `gh`.
7. Record only the PR number and base/head branch names, not the PR URL.
8. Do not merge the PR. Continue to a downstream edge only after predecessor changes
   are actually merged into the branch that the next edge must audit.

The helper command is:

```bash
python .agents/skills/vllm-metax-branch-port/scripts/publish_branch_port.py publish \
  --repo . --run-dir <run-dir> --target-ref master --pr-branch <dedicated-branch> \
  --title "<PR title>" --body-file <PR-body.md>
```

## Idempotency and failure

- Check authentication before the first push to avoid an orphaned remote branch.
- A remote PR branch at the same SHA may be reused.
- A remote PR branch at a different SHA is a conflict; do not force, overwrite or rebase
  it automatically.
- Reuse an existing open PR for the same base/head pair. Do not create a duplicate.
- Gerrit pushes reuse the local `Change-Id` to update an existing change. The helper
  must reject missing or duplicate Change-Ids in the selected commit range.
- Preserve redacted push/PR diagnostics under the run directory.

Publication completion means `PUSHED` for Gerrit or `PR_OPEN` for GitHub. It does not
mean submitted, approved or merged.

## Local-only master

When no GitHub remote owns master:

1. Create and validate the master commits locally.
2. Use the approved porting developer as author and committer and omit `Source:`.
3. Record `publication_status: local-only` and
   `local_only_reason: master-is-not-a-github-remote`.
4. Do not push, fork, create a PR or emit a repository URL.
