# Project skills and agents

Project workflows are stored in `.agents/` and can be read by any coding agent.
Use this file as the discovery entry point when the client does not automatically
discover `.agents/skills/`. Read the selected `SKILL.md` and follow its linked
references; load only workflows relevant to the user's request.

| Task | Skill |
| --- | --- |
| Full upgrade or coordinated component upgrade | [vllm-metax-upgrade](.agents/skills/vllm-metax-upgrade/SKILL.md) |
| Upgrade environment preparation | [environment-check](.agents/skills/environment-check/SKILL.md) |
| Shared upgrade rules and tooling | [common-sense](.agents/skills/common-sense/SKILL.md) |
| New or standalone model adaptation | [model-adaptation](.agents/skills/model-adaptation/SKILL.md) |
| Model compatibility | [model-upgrade](.agents/skills/model-upgrade/SKILL.md) |
| Attention backends and cache contracts | [attention-upgrade](.agents/skills/attention-upgrade/SKILL.md) |
| Native operators, csrc and build contracts | [csrc-upgrade](.agents/skills/csrc-upgrade/SKILL.md) |
| Registry, quantization and CustomOps | [registry-upgrade](.agents/skills/registry-upgrade/SKILL.md) |
| Monkey patches in `vllm_metax/patch/` | [patch-upgrade](.agents/skills/patch-upgrade/SKILL.md) |
| Reduced dummy or real-weight checkpoints | [vllm-metax-model-trim](.agents/skills/vllm-metax-model-trim/SKILL.md) |

The portable coordinator role is
[vllm-metax-upgrade](.agents/agents/vllm-metax-upgrade.md). Read it when acting as
the upgrade coordinator. Clients with native subagent support may register this
role in their own configuration; the Markdown file supplies its instructions.
Native agent registration formats differ between clients.

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

## Running linters

```bash
# Run all pre-commit hooks on staged files:
pre-commit run

# Run on all files:
pre-commit run --all-files

# Run a specific hook:
pre-commit run ruff-check --all-files
```

Final upgrade acceptance requires `pre-commit run --all-files` to pass on the final
code. Fix all current hook failures, including pre-existing formatting issues, and
rerun until it passes. If hooks modify source, refresh affected validation evidence
before declaring the upgrade complete.
