# Project skills and agents

Project workflows are stored in `.agents/` and can be read by any coding agent.
Use this file as the discovery entry point when the client does not automatically
discover `.agents/skills/`. Read the selected `SKILL.md` and follow its linked
references; load only workflows relevant to the user's request.

## Contribution and review responsibilities

- Before opening or updating a code review change, search the review system for
  open changes containing the issue number or short area keywords, and compare
  related recent commits and branches. If an equivalent change is already open,
  update or coordinate that change instead of opening a duplicate. If the
  approach is materially different, explain the difference.
- Do not submit a standalone change for a typo, isolated style fix or other
  mechanical cleanup. Bundle low-value cleanup with substantive work unless the
  user explicitly asks for the cleanup itself.
- A person must own and be able to explain the change end-to-end. AI tools may
  assist with the implementation and review, but they cannot be the accountable
  author. The human owner must review every changed line and run the required
  validation.
- Before opening or updating any pull request, including a draft, run
  `pre-commit run --all-files` against the final submitted tree. If hooks modify files,
  include those changes and rerun until the command passes with no uncommitted hook
  output. Record the exact command and passing result in the PR's validation evidence.
- State in the code review description that AI assistance was used. Include
  duplicate-work analysis when relevant, the validation commands and results,
  and model evaluation evidence when the change affects model output, accuracy
  or serving.
- Agent-created GitHub pull requests must use a completed copy of
  `.github/PULL_REQUEST_TEMPLATE.md`; free-form notes or `VALIDATION.md` are not
  acceptable substitutes. Prepare the filled body as `PR_BODY.md` in the run directory,
  check only statements that are true, and run the owning workflow's body validator
  before publication. The branch-port publisher enforces this template contract and
  reruns `pre-commit run --all-files` before the remote push.
- All PR descriptions must be written in English. Briefly state the validation method
  and provide concrete commands and results under `Validation`. Under `Details`, give a
  detailed explanation of the PR content: what changed, why it changed, the behavior,
  design decisions and tradeoffs.
- Public PR descriptions must not expose vendor/chip, private environment, internal
  path or credential details. A pending check must use only
  `- Pending verification: <affected behavior or claim>` under `Pending Verification`;
  do not disclose why verification was unavailable. Keep restricted evidence in private
  run state.
- User-visible changes must update the corresponding documentation in the same
  change. This includes CLI arguments, configuration fields, environment
  variables, supported models, hardware requirements, defaults and deprecations.
- If the requested work is duplicate or has no substantive purpose, stop and
  explain what is missing rather than producing a change to satisfy the request.

| Task | Skill |
| --- | --- |
| Independent staged/worktree/commit code review | [independent-review](.agents/skills/independent-review/SKILL.md) |
| Full upgrade or coordinated component upgrade | [vllm-metax-upgrade](.agents/skills/vllm-metax-upgrade/SKILL.md) |
| Cross-branch commit audit and source port | [vllm-metax-branch-port](.agents/skills/vllm-metax-branch-port/SKILL.md) |
| Independent branch-port adversarial review | [vllm-metax-branch-port-reviewer](.agents/agents/vllm-metax-branch-port-reviewer.md) |
| Scheduled daily branch port | [vllm-metax-branch-port-daily](.agents/agents/vllm-metax-branch-port-daily.md) |
| Upgrade environment preparation | [environment-check](.agents/skills/environment-check/SKILL.md) |
| Shared upgrade rules and tooling | [common-sense](.agents/skills/common-sense/SKILL.md) |
| New or standalone model adaptation | [model-adaptation](.agents/skills/model-adaptation/SKILL.md) |
| Model compatibility | [model-upgrade](.agents/skills/model-upgrade/SKILL.md) |
| Attention backends and cache contracts | [attention-upgrade](.agents/skills/attention-upgrade/SKILL.md) |
| Native operators, csrc and build contracts | [csrc-upgrade](.agents/skills/csrc-upgrade/SKILL.md) |
| Registry, quantization and CustomOps | [registry-upgrade](.agents/skills/registry-upgrade/SKILL.md) |
| Monkey patches in `vllm_metax/patch/` | [patch-upgrade](.agents/skills/patch-upgrade/SKILL.md) |
| Reduced dummy or real-weight checkpoints | [vllm-metax-model-trim](.agents/skills/vllm-metax-model-trim/SKILL.md) |

The portable coordinator roles are:

- [independent-reviewer](.agents/agents/independent-reviewer.md) for fresh-context,
  read-only review of staged, unstaged, commit or commit-range code.
- [vllm-metax-upgrade](.agents/agents/vllm-metax-upgrade.md) for full or coordinated
  component upgrades.
- [vllm-metax-branch-port](.agents/agents/vllm-metax-branch-port.md) for independent
  linear-matrix cross-branch commit audits and source ports.
- [vllm-metax-branch-port-reviewer](.agents/agents/vllm-metax-branch-port-reviewer.md)
  for independent, read-only adversarial review of the recorded worktree, target
  applicability, behavioral evidence and post-rebase patch necessity.
- [vllm-metax-branch-port-daily](.agents/agents/vllm-metax-branch-port-daily.md)
  for idempotent scheduled runs with pre-approved scope, run-scoped worktrees and
  automatic fresh-context applicability/final reviews.

Clients with native subagent support may register these roles in their own
configuration; the Markdown files supply their instructions. Native agent registration
formats differ between clients.

Skills follow the `SKILL.md`, `references/`, `scripts/`, and optional `assets/`
layout. Existing `agents/openai.yaml` files are optional client metadata; the
workflow is defined by `SKILL.md`. Put reusable workflow helpers and their tests
inside the owning skill; keep general project tools and generated model examples
under `tools/`. Shared upgrade helpers belong to
`.agents/skills/common-sense/scripts/metax_upgrade/`, with their CLI
guide in [tooling.md](.agents/skills/common-sense/references/tooling.md).

Resolve command paths from the repository root; Markdown links are relative to
their containing file. Upgrade skills require their documented environment gate
for compatibility work. Maintaining these workflows or their tooling does not
itself start an upgrade or require that gate.

## Repository-local guidance

- Before editing a subtree, read the nearest `AGENTS.md` that applies to it.
  Then read the relevant project skill and its linked references. Do not start a
  domain-specific edit using only this root file.
- If repository instructions conflict with the requested change, stop and explain
  the conflict instead of silently choosing one rule.
- Changes to this file or the project skills must keep referenced paths valid and
  pass `markdownlint-cli2`. Do not weaken a validation gate to make an
  implementation pass.

## Development workflow

- Before installing or reinstalling vLLM or vLLM-MetaX from a checkout, remove
  all repository-local generated installation and packaging outputs from that
  checkout, including `build/`, `dist/`, `sdist/`, `.eggs/`, `*.egg-info/`,
  `cmake-build-*/`, `.deps/`, generated version metadata and stale wheel or
  source-distribution files. Do not rely on incremental install state to remove
  files generated by an older source version.
- Clean each framework checkout separately before its installation. Resolve and
  inspect the exact generated paths first, ensure they are inside the selected
  repository and are not tracked source, then remove only those paths. Do not use
  broad `git clean -fdx` cleanup, and do not remove the selected Python
  environment, model weights, user configuration or unrelated artifacts.
- Run Python, build and test commands against the interpreter selected for the
  supported MetaX image. Do not install dependencies into or validate the project
  with a host system Python. Apply
  [environment-check](.agents/skills/environment-check/SKILL.md) when preparing a
  compatibility upgrade.
- Both `python -m pip` and `uv pip --python <selected-python>` are supported entry
  points. Do not mix packages from different virtual or Conda environments.
- For native operators, csrc and build-contract work, read
  [csrc-upgrade](.agents/skills/csrc-upgrade/SKILL.md). Keep the native kernel
  provider and `USE_PRECOMPILED_KERNEL` setting consistent between build and
  runtime; CUDA-only build assumptions do not apply to MACA.
- Preserve the target-upstream organization and style wherever a MetaX difference
  is not required. Add `NOTE(metax):` comments only for necessary platform
  differences, and put substantial design rationale in the model specification
  or durable review record rather than inline code.

### Coding style

- Match the surrounding code style and minimize comments. Prefer legible,
  self-documenting code, and keep required comments and docstrings brief and
  direct. Assume the reader is familiar with vLLM and MetaX.
- Let Ruff and `pre-commit` determine Python formatting. The configured Ruff
  policy ignores `E501`, so do not introduce unrelated rewrites solely to enforce
  an arbitrary line width.
- Use Google-style docstring sections (`Args:`, `Returns:` and `Raises:`) where
  those sections are needed. Do not add reStructuredText fields such as
  `:param:` or `:return:`.

### Commit attribution

The commit-message hook adds `Signed-off-by`. For AI-assisted work, also use an
`Assisted-by:` trailer when available; otherwise disclose the assistance in the
code review description.

## Running linters

Before committing in a checkout, install the configured Git hooks:

```bash
pre-commit install --hook-type pre-commit --hook-type commit-msg
```

The `pre-commit` hook validates changed files, and the `commit-msg` hook adds the
required sign-off trailer.

```bash
# Run all pre-commit hooks on staged files:
pre-commit run

# Run on all files:
pre-commit run --all-files

# Run a specific hook:
pre-commit run ruff-check --all-files
```

Run the relevant hooks on every change. Every pull request, including an update to an
open PR, requires `pre-commit run --all-files` to pass on the final submitted tree. Fix
all hook failures reported by that run and rerun until it passes. If hooks modify source,
include the changes in the submitted revision and refresh affected validation evidence.
