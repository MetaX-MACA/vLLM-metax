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
4. Rebase or otherwise bind the reviewed patch to the latest frozen target baseline and
   obtain a validated `ADVERSARIAL-FINAL.json` with `ENDORSE_PORT` for every published
   intent. A pending or non-endorsing review blocks the push.
5. Push without force:

   ```bash
   git push --porcelain <logical-remote> HEAD:refs/for/<mx-target-branch>
   ```

6. Record `publication_status: pushed`, the logical remote, target branch, review ref and
   adversarial-review result. Do not record the remote URL.
7. Do not submit the change. Human review and submission remain required.

The helper command is:

```bash
python .agents/skills/vllm-metax-branch-port/scripts/publish_branch_port.py publish \
  --repo . --run-dir <run-dir> --target-ref <mx-target-branch> \
  --worktree-json <run-dir>/WORKTREE.json \
  --adversarial-review <run-dir>/ADVERSARIAL-FINAL.json
```

## GitHub PR description

Every agent-created GitHub PR uses a completed copy of
[`.github/PULL_REQUEST_TEMPLATE.md`](../../../../.github/PULL_REQUEST_TEMPLATE.md) and
is written entirely in English. A free-form release note or `VALIDATION.md` is not a
substitute for the PR template.

1. Copy the repository template to `<run-dir>/PR_BODY.md`.
2. Fill `Overview`, `Claims`, `Validation` and `Details` with confirmed facts. Keep the
   overview to 1-2 lines and link relevant issues, pull requests and duplicate-work
   analysis.
3. Under `Validation Method`, briefly state how the change was validated and what
   behavior the checks cover. Detailed explanation belongs under `Details`.
4. Under `Validation Evidence`, list exact commands and results. Include
   `pre-commit run --all-files` and its passing result. Use public, reproducible
   configuration only, do not expose vendor/chip or private environment details and do
   not state unmeasured performance or correctness claims.
5. Under `Pending Verification`, write `None.` when complete. For a check that remains
   pending, use only `- Pending verification: <affected behavior or claim>`. Do not
   disclose why the check is unavailable or include vendor/chip, private package,
   proprietary
   environment, internal path or credential details.
6. Under `Details`, provide a detailed explanation of the PR content: what changed,
   motivation, design decisions, root cause, behavior, tradeoffs, limitations and
   documentation updates. State the target-first applicability evidence: target
   counterpart, why an existing equivalent does not cover the gap, source assumptions
   that remain valid and any deliberate deviation from the source mechanism.
7. For a source port, `Details` must include the complete source-to-target commit SHA
   mapping and identify every pending review risk. Do not include repository URLs or
   credentials.
8. Validate `ADVERSARIAL-FINAL.json` against the latest target baseline and final head.
   Publication requires `ENDORSE_PORT` for every selected intent and actual
   fail-before/pass-after behavior evidence, except for a specifically approved
   non-runtime validation contract.
9. Mark a checklist item only after it is true. Agent-created PRs must disclose AI
   assistance. If any required item cannot truthfully be checked, block publication
   instead of weakening the checklist.
10. Validate the completed body before publication:

    ```bash
    python .agents/skills/vllm-metax-branch-port/scripts/publish_branch_port.py \
      validate-pr-body --repo . --body-file <run-dir>/PR_BODY.md

    python .agents/skills/vllm-metax-branch-port/scripts/check_adversarial_review.py \
      --review <run-dir>/ADVERSARIAL-FINAL.json --stage final \
      --expected-target-sha <master-sha> --expected-head-sha <patch-head-sha> \
      --worktree <metax-worktree> --require-endorsement
    ```

The PR-body validator requires English prose, every section and validation subsection
from the repository template, non-placeholder content, a passing
`pre-commit run --all-files` report, safe pending-check entries and every template
checklist item checked. The adversarial-review validator requires independent review,
target-first evidence, checked assumptions, post-rebase freshness and endorsement for
every published intent.

## GitHub master publication

GitHub publication uses a dedicated non-force branch and a PR whose base is `master`.
Never push to `master` directly and never merge the PR automatically.

Use this section only when the confirmed logical GitHub remote owns master. Otherwise
follow the local-only rule below.

1. Verify that `master` passes the version gate.
2. Verify the GitHub remote and a working authenticated `gh` CLI before pushing.
3. Use the approved porting developer as both author and committer for every master
   commit. Omit `Source:` trailers.
4. Rebase onto the latest master, repeat target-first applicability review and obtain a
   validated `ADVERSARIAL-FINAL.json` with `ENDORSE_PORT` for every published intent. The
   artifact must bind `review_worktree` and `worktree_branch` to the clean checkout being
   published.
5. Run `pre-commit run --all-files` on the final tree. If hooks modify files, commit or
   otherwise include those changes as appropriate and rerun until the command passes
   without producing further changes. The helper runs the command again before the
   remote push and blocks if it fails or modifies the worktree.
6. Complete and validate `<run-dir>/PR_BODY.md` using the rules above. Record the exact
   pre-commit command and passing result in `Validation Evidence`. Use a concise title
   consistent with the change and the repository commit style.
7. Push a dedicated branch without force:

   ```bash
   git push --porcelain <github-remote> HEAD:refs/heads/<pr-branch>
   ```

8. Search for an open PR with the same head branch and base `master`.
9. Reuse the open PR when present. Otherwise create one with `gh`. The helper validates
   the supplied body before pushing. On reuse, it validates the existing body and updates
   a nonconforming body from `PR_BODY.md`.
10. Record only the PR number and base/head branch names, not the PR URL.
11. Do not merge the PR. Continue to a downstream edge only after predecessor changes
    are actually merged into the branch that the next edge must audit.

The helper command is:

```bash
python .agents/skills/vllm-metax-branch-port/scripts/publish_branch_port.py publish \
  --repo . --run-dir <run-dir> --target-ref master --pr-branch <dedicated-branch> \
  --title "<PR title>" --body-file <run-dir>/PR_BODY.md \
  --worktree-json <run-dir>/WORKTREE.json \
  --adversarial-review <run-dir>/ADVERSARIAL-FINAL.json
```

## Idempotency and failure

- Check authentication before the first push to avoid an orphaned remote branch.
- A missing, expired or mismatched `WORKTREE.json` lease blocks both Gerrit and GitHub
  publication.
- A missing, stale or non-endorsing `ADVERSARIAL-FINAL.json` blocks both Gerrit and
  GitHub publication.
- A remote PR branch at the same SHA may be reused.
- A remote PR branch at a different SHA is a conflict; do not force, overwrite or rebase
  it automatically.
- Reuse an existing open PR for the same base/head pair. Do not create a duplicate.
- A reused PR whose body does not satisfy the repository template is updated from the
  validated `PR_BODY.md`; record this in `PUBLISH.json` through `pr_body_updated`.
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
