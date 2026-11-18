# Review modes and artifact

## Review context

`review_context.py snapshot` writes:

```json
{
  "schema_version": 1,
  "review_mode": "staged",
  "repo": "/absolute/repo",
  "worktree": "/absolute/repo",
  "branch": "feature/example",
  "base_sha": "<full-sha-or-null>",
  "head_sha": "<full-sha-or-null>",
  "index_tree_sha": "<tree-sha-for-staged-else-null>",
  "review_hash": "<sha256-of-exact-reviewed-target>",
  "changed_files": ["path"],
  "untracked_files": [],
  "created_at_utc": "<timestamp>"
}
```

### staged

- Reviews `git diff --cached`.
- Binds to the index tree and staged diff hash.
- Untracked files are not reviewed unless staged.

### worktree

- Reviews `git diff` plus untracked-file content.
- Binds to the unstaged diff and untracked manifest hash.

### commit

- Reviews the diff from an explicit base to one commit.
- Binds to the full base/head SHAs and diff hash.

### range

- Reviews the complete diff from base to head.
- Binds to full SHAs and diff hash.

## Review artifact

```json
{
  "schema_version": 1,
  "review_mode": "staged",
  "review_context_sha256": "<sha256-of-context-json>",
  "review_hash": "<copy-from-context>",
  "reviewer_mode": "independent-agent",
  "reviewer_identity": "<fresh-task-or-person-id>",
  "implementer_identity": "<implementer>",
  "reviewed_at_utc": "<timestamp>",
  "status": "APPROVE",
  "summary": "<concise overall result>",
  "findings": [
    {
      "severity": "blocking",
      "path": "path/to/file.py",
      "line": 42,
      "summary": "<concrete defect>",
      "evidence": "<why the defect exists>",
      "required_action": "<specific fix>"
    }
  ],
  "checks": [
    {
      "name": "<review or validation check>",
      "status": "PASSED",
      "evidence": "<command/result or concrete code evidence>"
    }
  ]
}
```

Allowed finding severities: `blocking`, `major`, `minor`, `info`.

Allowed check statuses: `PASSED`, `FAILED`, `NOT_RUN`, `NOT_APPLICABLE`.

An `APPROVE` result cannot contain `blocking` or `major` findings. A
`CHANGES_REQUIRED` result must contain at least one `blocking` or `major` finding.
