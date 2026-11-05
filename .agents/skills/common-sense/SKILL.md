---
name: common-sense
description: Shared rules and state tools for vLLM-MetaX compatibility upgrades, including upstream-first ports, complete file ownership, environment approval, phased validation and recovery. Required by the upgrade coordinator and domain upgrade skills; not a general-purpose advice skill.
---

# Shared vLLM-MetaX Upgrade Rules

Apply this workflow with the selected domain skill. Domain skills contain only their
special rules; the coordinator owns full-repository scheduling and completion.
Maintaining these skills or their tooling does not start an actual compatibility upgrade.

## Scope and environment

Identify the requested source snapshot, target revision, checkout to edit and validation
scope. Preserve user edits. Locate and resume the existing durable run before creating
another baseline. Full upgrades use `--mode full`; standalone domain work uses component
mode and makes no whole-framework completion claim. A specialist participating in a
full run must use that run and its phase gates, not create a component run to bypass them.

Apply [environment-check](../environment-check/SKILL.md) for target/Python/checkouts,
fresh independent inventory and all environment preparation. Collect the complete pending
list and obtain **one explicit approval of the preparation plan** before mutations. It
covers conditional framework reinstalls and ordinary dependency changes with image locks.
After setup, machine verification binds the actual environment to that approved plan;
do not request a second snapshot confirmation or invent a user reply.

**Both frameworks must install successfully and pass preparation verification before
compatibility decisions, adaptation edits or adaptation tests.** Installation failure
blocks adaptation, including source changes intended to fix that installation. Preserve
logs and continue only permitted environment diagnostics/retries. Out-of-plan changes
are blockers; do not repeatedly prompt or expand authorization during a long run.
Fresh unchanged facts retain the existing plan verification across specialists/resume.

## Upstream-first source adaptation

Use the exact confirmed target implementation, following the corresponding CUDA path
and base framework API. Understand the previous MetaX changes and their MACA reasons,
then reapply only necessary changes onto the new upstream code. Inspect the complete
touched module, not just changed symbols or an old adaptation diff.

- Preserve upstream module/function organization, names, signatures, decorators,
  control flow, code style, **blank lines and comments** wherever the MACA adaptation
  does not require a change. Do not refactor, reorder, extract helpers or reformat the
  upstream implementation. Prefer small additions or removal of an inapplicable feature.
- Preserve explanatory upstream comments. Add concise nearby `NOTE(metax):` reasons
  for necessary differences; avoid redundant commentary or large new explanations in
  code. Put substantial design differences in the model spec or durable review record.
- Compare the complete result with the exact target, justify every remaining difference
  including retained differences, and check explicit user settings and caller-owned state.
- Inspect every relevant upstream commit across the known upgrade range, including its
  other changed production files and direct consumers. Record moved, local-only,
  target-only, unchanged and removed units; a final tree diff alone misses semantics.

## Complete inventory, dependencies and ownership

Before adaptation, initialize the complete file inventory in the run. Perform and
record read-only global dependency/model-contract analysis with `analysis-put` before
platform edits. Cover every file and every baseline model, required framework interfaces,
cache layouts, quantization contracts, dependency edges and coordinated cycle handling.
Expected port edits do not invalidate the plan; new files/models or a changed confirmed
environment require updating it. It covers **every
Python file under `vllm_metax/`**, including initializers, third-party code, inactive paths,
indirect dependencies and added files, **every csrc source/header and native build
input**, plus production/build/ABI/requirements files. Native sources require full upgrade
under both mcoplib and source-kernel modes; apply [csrc-upgrade](../csrc-upgrade/SKILL.md).
Keep deleted files as inventory entries with reviewed removal dispositions. No-edit
files need a reviewed retain decision, not an automatic completed mark.

Assign each file a phase, owning skill, task/configuration, exact target/prior mapping,
MACA differences, disposition and acceptance obligations before editing. Trace imports,
startup side effects, registrations, calls and data/state contracts recursively to a
verified interface or installed component boundary. Update required **indirect local
dependencies** together: a registry port is incomplete if its compressed-tensors MoE
path still selects an unreviewed local oracle, experts implementation or modular kernel.
Directory membership is not a dependency boundary. Give shared files one owner and
phase; split configuration checks into separate tasks with that same ownership.

Record dtype/layout/stride/alignment/index/scale semantics, ownership, argument/default
meanings, returns and lifecycle where relevant. Adapt cyclic interfaces as a coordinated
unit. Read-only inspection of later phases is allowed. In a full run, implement a
required indirect dependency with its consuming phase; do not postpone it just because
its directory would normally belong to a later framework phase. Model-private code
remains in the models phase. Do not semantically import unfinished MetaX code into the
coordinator; candidate execution belongs in fresh subprocesses with declared boundaries.

## Adaptation and validation are separate

Read [validation.md](references/validation.md) and
[hardware-validation.md](references/hardware-validation.md) before planning checks.
Full runs follow the coordinator's ordered stages. During source adaptation, permit
static/review checks, framework/operation unit tests, component numerical checks and
model-free startup/dispatch closures. **Do not construct or load a complete model,
run dummy/trimmed-model smoke, generate tokens, or run model e2e/batched_test until
all framework, model and patch source adaptation is complete and the run reaches
final validation.** Label execution honestly; model construction is not a component
check merely because its command is called a unit test.

Persist mappings and local acceptance before marking source `implemented`. Use
`checks-put` to classify semantic changes and justify discriminating cases. API/dispatch/
lifecycle changes require component contract tests; layout/quantization/indexing/numerical/
communication changes also require real component numerical evidence. Static/review alone
are sufficient only for a justified no-semantic-change task. Unsupported-chip numerical
branches can defer only against a matching explicit unverified coverage record; supported
branches still need their checks. Full model construction remains in the last phase. This means
source and required local checks are ready for the next phase; it does not mean final
verification. Every inventory file displays one of:

| File status | Meaning |
| --- | --- |
| 未适配 | No adaptation has started. |
| 正在适配中 | Shows the owning skill; may be source-adapted and awaiting final evidence. |
| 适配完成并验证完毕 | All required current evidence for the declared configuration is complete. |

Keep the separate `source_ready` field visible. Full completion requires every inventory
entry in the third state, all phase gates passed, required model e2e **and batched
inference correctness** evidence, and a passing `pre-commit run --all-files` on the
final code. Fix all current hook failures, including pre-existing formatting issues;
follow the final lint procedure in [validation.md](references/validation.md). Completion is limited to the declared current-chip
validation scope. List unavailable quantization branches separately as **unverified**;
they still need source adaptation and must not be reported as supported or globally
verified. The user permits completion within the available chip's supported scope.

## Durable progress and recovery

Read [run-state.md](references/run-state.md) and [tooling.md](references/tooling.md).
Use [upgrade.py](scripts/metax_upgrade/upgrade.py) for authoritative state, generated
`TODO.md`, `FILES.json` and `FILES.md`. Persist ownership, mappings, dependencies,
contracts and next actions before edits; checkpoint findings and results before a handoff
or interruption. Do not hand-edit completion flags or replace an old baseline.

Source, relevant contracts, environment, component identities, check inputs or workflow
changes invalidate affected evidence and transitive consumers. Contracts bind only their
producer/consumers. Source acceptance has its own signature; changing model test-matrix
rows invalidates final model evidence without revoking unrelated source readiness. On resume, reconcile current
hashes, partial diffs, interrupted processes and approvals before continuing. Retain
failed/stale evidence. Required blocked checks keep completion pending; independent
source work can continue only while the mandatory installation/environment gate is ready.
Report concrete changes, verified scope, unverified hardware branches, remaining obligations and exact run/TODO/inventory paths.

## Baseline model acceptance

Every `tools/models/**/config.json` found when initializing/resuming a run is a mandatory
model sample, tracked in `MODELS.json` and TODO. Bind each sample to a spec in its local
implementation directory, a `model-review` source task and exact e2e/batched test rows
using `model-bind`; the row identifies `model_sample` and the actual sample `model_path`.
Multiple configurations of one architecture are distinct mandatory samples.

Only GPU-count or VRAM shortages may use `model-skip`, in final validation, with required
and available resources, a concrete explanation and a saved device/memory/OOM artifact.
Skips bind the current sample/source/config/environment and become stale when these or
the evidence change. The model remains **skipped**, never passed; final acceptance lists
all skips. Other execution defects, missing weights, missing specs or missing bindings
remain blockers. Unsupported quantization branches retain their separately agreed chip
scope and unverified record; they are not silently removed from the baseline sample set.

Apply [model-adaptation](../model-adaptation/SKILL.md) for concrete-model isolation,
per-model specs and the import audit in both new model work and model upgrades.

For baseline samples using dummy weights, acceptance is explicitly `dummy_smoke`:
startup, valid prefill/decode generation and concurrent request/batch stability, without
semantic accuracy claims. Real-weight configurations use `accuracy` and their semantic
cases. Declare `configuration.validation_mode` in model rows and `load_format: dummy`
where applicable. A dummy smoke pass cannot satisfy real-weight accuracy acceptance.
