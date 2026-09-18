---
name: vllm-metax-patch-upgrade
description: Audit and adapt vLLM-MetaX monkey patches against a target upstream revision. Decide whether to retain, update, migrate, or remove each patch; maintain patch headers and algorithm explanations; validate changes and record the evidence. Use for patch compatibility reviews after vLLM, Transformers, Torch, or Triton upgrades, not for unrelated model implementations or general refactoring.
---

# vLLM-MetaX Patch Upgrade

Preserve necessary MetaX behavior while inheriting the target upstream implementation.
Deliver reviewable changes, a complete patch inventory with decisions, and accurate
validation results. Explain why each difference exists and when it can be removed.

## 1. Establish the baseline and repository requirements

- Read applicable `AGENTS.md` files, `vllm_metax/patch/README.md`, patch templates,
  and existing audit records. Follow current repository requirements and user instructions.
- Inspect working-tree and staged changes; preserve existing work. By default, do
  not commit, stage changes, alter installed dependencies, or modify the upstream checkout.
- Complete the mandatory [environment preflight](references/environment-preflight.md)
  before making compatibility decisions. Run the bundled read-only probe with the
  selected interpreter and explicit local checkout paths; inspect its source comparisons.
- Establish all five locations: Python/venv, local vLLM source, installed vLLM package,
  local vllm_metax source, and installed vllm_metax package. Separately establish the
  effective import origin of each package under the actual test/launch conditions.
- Present the detected environment and proposed comparison/runtime plan, then proactively
  ask the user to confirm them before compatibility decisions, patch edits, or runtime
  validation. Follow the confirmation procedure in the preflight reference. Reuse an
  explicit confirmation already given in this session when the mapping is unchanged;
  a later instruction to proceed without confirmation takes precedence.
- Use the user-specified interpreter. For a uv environment at `/opt/venv`, run Python
  checks and tests with `/opt/venv/bin/python`, not system or Conda Python. An activated
  shell, matching version strings, a wheel filename, or `direct_url.json` alone does
  not establish source/runtime correspondence.
- Resolve discrepancies by identifying the intended comparison source and validation
  runtime. Continue independent inventory work, but do not base compatibility decisions
  on an unverified checkout or claim local edits were tested when imports resolve to a
  different copy. Include unresolved target choices in the environment confirmation;
  do not reinstall,
  change environment variables globally, or switch checkouts to conceal a mismatch.
- Distinguish adapting to the installed version from upgrading to remote HEAD.
  Unless the user requests the remote latest version, use the established target
  revision and state it clearly. Do not call a local checkout the latest remote version.

## 2. Inventory every patch and its activation path

Start from the directory's file list and trace plugin entry points and package
initializers. Search beyond `@patch`: include direct attribute assignments, registry
mutations, `sys.modules` redirects, imported aliases, and disabled or unimported patches.

For each patch, record its file, target symbol, purpose, activation status, upstream
location, MetaX differences, decision, evidence, and validation approach. A single
file may contain patches with different decisions. Classify templates, shared helpers,
and aggregate import modules separately as infrastructure.

## 3. Compare upstream behavior and decide per patch

Read the original explanation, complete target implementation, relevant helpers, and
callers before deciding. AST comparisons can help with signatures and function bodies.
Trace dynamic exports, inherited methods, and imported aliases to their definitions;
a failed static lookup does not establish that an upstream target was removed.

| Decision | Evidence and action |
| --- | --- |
| Remove | Upstream fixes the issue, the affected path no longer exists, or an existing hook/redirect already provides the behavior. Verify callers, then remove the implementation and its activation entries. |
| Migrate | A registry, platform hook, CustomOp, or narrower extension point now supports the requirement. Move to it, verify dispatch, and remove the redundant monkey patch. |
| Update | MetaX behavior is still necessary, but upstream signatures, helpers, branches, or contracts changed. Reapply only the required differences to the current implementation. |
| Retain | The patch remains compatible and addresses a concrete platform, hardware, checkpoint, or product requirement. Record the reason and removal condition. |

- Age, code similarity, disabled status, or an existing PR link alone is not evidence
  for removal. Inspect history when needed and confirm the fix exists in the target revision.
- Investigate why upstream deliberately removed old behavior before restoring it
  from a patch's copied implementation.
- Separate hardware constraints from software bugs. Without suitable GPU or distributed
  evidence, do not declare shared-memory, warp, IPC ABI, or communication restrictions
  resolved. Record the reason for retention and the remaining validation need.
- Verify behavioral claims in comments against code. For example, a stable sort over
  misplaced requests does not necessarily preserve order across an entire region.

## 4. Make the smallest complete adaptation

- Prefer existing upstream registration or extension points. A small wrapper should
  delegate unaffected behavior to the original implementation and preserve explicit inputs.
- When a full replacement is necessary, preserve the target upstream name, signature,
  return contract, decorators, and unchanged code. Mark only the required MetaX
  differences. Do not let an old function copy suppress new validation, model support,
  or resource cleanup.
- Preserve descriptor semantics, Triton decorator order, lazy imports, and initialization
  timing. Check whether `from ... import ...` already bound the old object; successful
  patch import does not prove every caller uses the replacement.
- Follow repository templates for `@patch`. Use `allow_missing=True` only when adding
  an intentional compatibility attribute, never to hide renamed targets or typos.
- Inspect argument semantics rather than forwarding by name alone. For example,
  `scale=None` may select dynamic quantization while a supplied scale selects static quantization.
- Add or update platform-specific registry entries without replacing the entire table.
  Compatibility defaults must preserve explicit user settings. Avoid unnecessary mutation
  of caller-owned dictionaries or configuration objects.
- On removal, check references, import order, and overlapping replacements. Keep unrelated
  cleanup outside the task.

For quantization, tokenizer, batch ordering, registration, allocator, or kernel patches,
consult [review-patterns.md](references/review-patterns.md) as needed. Its examples guide
inspection; they are not fixed decisions for future upstream versions.

## 5. Treat explanatory comments as part of the deliverable

Follow the current patch README. Every Python file in the patch directory, including
initializers and shared utilities, should have the dedicated header. Preserve license
and copyright notices. Templates may retain fill-in placeholders; active files must
contain meaningful descriptions.

```python
# -----------------------------------------------------------------------------
# Note: Describe the concrete issue, trigger, and reason for the MetaX difference.
#
# Affected versions: State evidence-backed affected versions and the reviewed revision.
#
# Remove at: Give a verifiable condition, such as upstream support or a resolved limit.
# -----------------------------------------------------------------------------
```

- Do not substitute scattered `Verified against`, `Remove when`, or docstrings for
  these fields. Describe infrastructure honestly without inventing an upstream defect.
- Preserve existing algorithm explanations, examples, parameter descriptions, and return
  semantics. Update them when implementation changes; do not delete useful explanations
  merely to shorten the file.
- For complex sorting or kernel patches, explain input/output contracts, invariants,
  key indices and mappings, the rationale, and the actual differences from upstream.
- Distinguish correctness requirements, platform restrictions, performance policy, and
  implementation choices. Preserving non-decode request order does not mean one specific
  request must precede another for numerical correctness.
- Include a discriminating input/output example for boundary behavior. Do not promise
  unmeasured performance gains. Match the source language and write for future maintainers.

## 6. Validate according to the change

- Run relevant lint, formatting, syntax, and diff checks. For comment/docstring-only
  changes, compare ASTs after removing docstrings to confirm executable behavior is unchanged;
  a full model suite is unnecessary.
- For behavioral changes, choose the least expensive tests that expose real regressions:
  boundary arguments, explicit settings, nested configurations, serialization, request/state
  alignment during reordering, and preserved upstream behavior.
- Run import smoke tests in fresh processes to avoid duplicate-patch failures. Verify
  both successful target import and that actual callers reach the replacement.
- Recheck environment correspondence after dependency installation, checkout changes,
  or changes to the interpreter, working directory, or import path. In the actual test
  process, record package `__file__`/`__path__`, relevant extension-module paths, and
  the active MetaX plugin origin; compare them with the preflight plan.
- For quantization, attention, or kernel changes, compare actual GPU output with a reference
  when the environment permits. Cover relevant boundary lengths, dtypes, layouts, and masks.
  Performance claims require benchmarks; communication claims require suitable distributed runs.
- Reuse nearby tests. If the root conftest introduces unrelated dependencies, an isolated
  test directory may use `--confcutdir`; state which fixtures this bypasses.
- Separate patch failures from environment ABI or optional-dependency failures. A narrowly
  scoped, process-local workaround may isolate unrelated paths. Do not silently modify the
  environment or report isolated tests as complete integration validation.
- Record the interpreter, commands, results, temporary workarounds, and untested scope.
  Mocked dispatch tests do not replace real kernel numerical validation.

## 7. Finish the audit and handoff

Update the existing `AUDIT.md` or repository-designated review record. Account for every
inventory entry and ensure decisions match the final code. Include:

- The environment correspondence table: interpreter/venv, both local checkouts and
  commits/dirty state, both installed distributions and package locations, effective
  import origins, content-comparison results, and any intentional source/wheel differences.
- Each retain/update/migrate/remove decision, concrete upstream evidence, and removal condition.
- Important behavioral differences, reproducible validation commands, results, and untested scope.

Conclude with the completed changes, key behavior changes, validation results, and audit
location. If the environment blocks validation, state what was completed and what remains
unverified. Do not equate source inspection or partial success with full model validation.
Once appropriate checks pass, expand testing only for new changes, failures, or unresolved concerns.
