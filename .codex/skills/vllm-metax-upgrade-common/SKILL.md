---
name: vllm-metax-upgrade-common
description: Establish the shared environment, source/runtime correspondence, target confirmation and validation evidence for MetaX vLLM upgrades. Use as the common prerequisite of the patch, attention, registry and model upgrade skills, or for an explicitly requested MetaX environment preflight; it does not perform domain-specific adaptations.
---

# vLLM-MetaX Upgrade Common

Provide one shared baseline for specialist upgrade workflows. Read this skill when a
calling skill requires it; apply only the common checks here, then continue that
skill's domain workflow. Do not activate other specialist skills automatically.

## Environment and target

Run the read-only [environment preflight](references/environment-preflight.md) using
this skill's `scripts/probe_environment.py` and the user-selected interpreter. It owns
the five-location mapping, component inventory, content comparison and effective
runtime-origin checks. Specialist skills must not copy the probe or maintain their own
preflight instructions. Examples use `/opt/venv/bin/python`; discover actual paths.

**Proactively confirm the comparison/runtime mapping once if it has not already been
established in this session.** Reuse unchanged confirmation across all calling skills;
honor an explicit instruction to skip the question. A switch from patch to model work
alone does not require confirmation again. Expected local edits do not invalidate the
mapping. Ask again only when the intended interpreter, target or runtime mapping changes
or becomes uncertain. If this requirement causes a pause, link this SKILL.md, quote
this instruction and explain the concrete unresolved mapping.

Continue independent inventory while awaiting a required target choice. Do not treat
silence, elapsed time or successful probe execution as confirmation. This is target
clarification, not renewed permission for already authorized work.

Keep one reusable task record containing:

- Selected interpreter/venv and runtime cwd.
- Local vLLM target revision, installed vLLM identity and relevant content differences.
- Local MetaX snapshot (index/worktree/revision), installed identity and effective imports.
- Component versions, selected callable/extension origins and device/ABI evidence.
- Intended comparison source, edited checkout and actual validation imports.
- Confirmed choices or the user's instruction to skip confirmation; remaining unknowns.

Use the established target, not assumed remote HEAD. Version strings alone cannot
prove correspondence. The probe is read-only and standard-library-only; exit zero
means collection succeeded, not that APIs, binaries or numerical behavior are valid.

## Shared scope and verification rules

- Preserve user changes. Keep reviews read-only. Do not stage, commit, reinstall,
  alter global import paths, edit upstream/site-packages or send external messages
  unless the task authorizes it. The calling skill sets the adaptation scope.
- Identify the exact snapshot. For staged reviews, export the index and prove that
  validation imports it; unstaged fixes must not satisfy staged dependencies silently.
- Trace actual plugin dispatch, wrappers and installed callables. Source comments,
  matching signatures and upstream capability declarations are hypotheses to verify.
- For now, generate self-contained temporary validation scripts from current contracts;
  do not depend on repository `tests/`, its fixtures/helpers/conftest or historical
  `/tmp` files. Select meaningful cases in the calling skill's validation guidance.
- Record commands, inputs, expected/observed behavior, imported sources, component
  identities and process-local workarounds. Isolated mocks test only the isolated
  contract; kernel tests do not establish full-model or distributed correctness.
- Distinguish source inspection, signature checking, runtime reproduction, numerical
  validation and untested scope. Preserve failures and unsupported paths in the report.
- Run relevant format/lint/type/diff checks and disclose unavailable checks. Recheck
  affected snapshots for concurrent edits before claiming findings are current.

This skill owns shared mechanics only. Patch headers/audits, attention index semantics,
registry quantization rules, model structure alignment and model-specific upstream-bug
handling stay in their specialist skills. Reusing common checks does not import one
specialist's policy into another. No separate audit artifact is required by this skill.
