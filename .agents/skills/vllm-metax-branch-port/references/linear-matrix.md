# Linear branch matrix

The matrix is a linear strong-guarantee policy, not a claim that Git ancestry contains
every source commit. New target commits are semantic carry-forward results with new
SHAs and new `Change-Id` values.

## Configured chain

```text
mx/v0.30.0-dev -> mx/v0.32.0-dev -> mx/v0.33.0-dev -> master
```

| Target edge | Target vLLM public version |
| --- | --- |
| `mx/v0.30.0-dev -> mx/v0.32.0-dev` | `0.32.0` |
| `mx/v0.32.0-dev -> mx/v0.33.0-dev` | `0.33.0` |
| `mx/v0.33.0-dev -> master` | `0.33.0` while master targets that upstream baseline |

The final edge is valid only while `master` targets upstream v0.33. The policy
intentionally skips `mx/v0.31.0-dev`; changes present only on that branch are outside
this chain's guarantee.

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
