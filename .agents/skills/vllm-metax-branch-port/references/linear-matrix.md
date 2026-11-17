# Linear branch matrix

The matrix is a linear strong-guarantee policy, not a claim that Git ancestry contains
every source commit. New target commits are semantic carry-forward results with new
SHAs and new `Change-Id` values.

## Dynamic chain

Resolve the matrix from actual Gerrit branch existence:

1. List `refs/heads/mx/v*-dev` on the confirmed Gerrit remote.
2. Keep only branches matching `mx/vX.Y.Z-dev`.
3. Sort them by semantic version.
4. Starting from the requested source version, use every later existing branch.
5. Append `master` as the final target.

A version exists only when its `mx/vX.Y.Z-dev` branch exists on the Gerrit remote.
There is no placeholder edge for an absent version. For example, if only v0.30.0 and
v0.32.0 exist:

```text
mx/v0.30.0-dev -> mx/v0.32.0-dev -> master
```

Absent v0.31.0 and v0.33.0 branches are skipped because those vLLM versions do not exist
in the branch matrix. If a branch appears later, the next matrix resolution includes it.
Read master's public version from `setup.py` at the confirmed GitHub or local master
baseline.

Discover branch existence with a live Gerrit query by default. If credentials prevent
that query, only a user-refreshed set of remote-tracking refs may be used as an explicit
fallback; record `branch_inventory_source`.

The final edge is valid only while the GitHub `master` public version is greater than or
equal to the latest available `mx/vX.Y.Z-dev` public version.

The user normally supplies only the starting branch and final target. Resolve the
intermediate path from the dynamically discovered chain, then display every node and
edge for confirmation before audit. A user-provided explicit chain is an override, not a
silent default. If the override omits an existing `mx/` branch inside the requested
version interval, display it as skipped for explicit confirmation.

Every run also records `final_target: master`. Closing an intermediate edge does not
complete the chain; continue only after its changes are merged into the actual target
branch. If a GitHub remote owns master, its public version must be greater than or equal
to every available branch matching `mx/vX.Y.Z-dev`. If no GitHub remote owns master,
master remains local-only and produces no push or PR.

Provider routing is by branch contract, not by remote name:

- `mx/vX.Y.Z-dev` uses the Gerrit remote and review ref `refs/for/<branch>`.
- A malformed `mx/` name is blocked.
- `master` uses a pull request only when the confirmed GitHub remote owns master.
- Master without a GitHub remote is local-only.
- No repository URL is a policy input or may appear in skill/run artifacts.

## Edge invariants

Each edge connects one frozen source snapshot to one frozen target baseline.

- Audit only non-merge commits in the direct source snapshot.
- Do not inspect older branches to search for previously missed commits.
- Every candidate must have a final disposition before the edge closes.
- A commit that is not carried forward creates no target commit and disappears from
  downstream candidate sets.
- Downstream edges see the actual target commit snapshot, including replacement and
  adapted implementations.
- A downstream edge may open only when every predecessor edge is closed through the
  latest predecessor source SHA.
- Any source advance or target rewrite after closure requires a new fetch, audit and
  matrix update before making another completeness claim.
- Remote publication is a separate recorded phase. Local closure, Gerrit publication and
  GitHub PR creation do not imply review submission or merge.

The guarantee is therefore:

> Every non-merge commit in each source snapshot is accounted for, and every commit
> visible downstream is part of the actual target snapshot produced by the preceding
> edge.

## Edge status

Use these states:

- `OPEN`: the edge has unprocessed source commits or nonterminal decisions.
- `IN_PROGRESS`: approved ports are being adapted or validated.
- `BLOCKED`: work cannot continue because of a concrete external fact, such as a
  missing branch, unresolved source-author confirmation or failed environment gate.
- `CLOSED`: every source non-merge commit has a terminal disposition, all approved
  ports are committed, validation is current, and the recorded source/target baselines
  are unchanged.
- `PUBLISHED`: the approved Gerrit changes were pushed to `refs/for`, or the approved
  GitHub PR was created/reused. This does not mean submitted or merged.

`PORT_REQUIRED` and `NEEDS_REVIEW` are nonterminal. `PORTED`, `ALREADY_PRESENT`,
`UPSTREAM_FIXED`, `VERSION_SPECIFIC`, `OBSOLETE` and explicitly approved
`OUT_OF_SCOPE` are terminal.

## Matrix record

Store the maintained state outside tracked source at
`.metax-branch-port/MATRIX.json`. Each edge should retain:

```json
{
  "id": "v0.30-to-v0.32",
  "source_ref": "mx/v0.30.0-dev",
  "target_ref": "mx/v0.32.0-dev",
  "vllm_ref": "<vllm-branch-or-tag>",
  "vllm_commit": "<full-vllm-sha>",
  "vllm_public_version": "<version>",
  "vllm_previous_release": "<previous-version>",
  "metax_public_version": "<same-public-version>",
  "source_watermark_sha": "<full-sha>",
  "target_baseline_sha": "<full-sha>",
  "fetched_at_utc": "<timestamp>",
  "predecessor_edge": null,
  "final_target": "master",
  "publish_requested": true,
  "publish_provider": "gerrit-or-github",
  "publish_remote": "<logical-remote-name>",
  "publication_status": "<not-requested/local-only/pushed/pr-open/blocked>",
  "status": "CLOSED",
  "environment_fingerprint": "<fingerprint-or-null>",
  "commits": []
}
```

For each source commit, retain:

```json
{
  "source_sha": "<full-sha>",
  "disposition": "PORTED",
  "target_sha": "<full-sha-or-null>",
  "target_change_id": "<new-git-change-id-or-null>",
  "original_author": "Name <email>",
  "evidence": "Concrete target behavior or superseding revision",
  "audit_artifact": "relative/path/to/AUDIT.md"
}
```

Use `target_sha: null` for terminal dispositions that create no target commit. Preserve
the complete source-to-target mapping and evidence; a branch name alone is not a stable
watermark.

Every edge records its explicit upstream vLLM mapping. `vllm_public_version` must equal
`metax_public_version`; local build suffixes do not permit a different public base.
Once the target upstream version changes, the old closure and environment fingerprint
cannot be reused as evidence for the new baseline.

For a GitHub master edge, store only the PR number and head/base branch names, never the
PR URL. Master commits use the approved porting developer as author and committer and
contain no `Source:` trailer; source traceability remains in this matrix.
