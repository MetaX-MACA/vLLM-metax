---
name: vllm-metax-upgrade
description: Coordinate a complete vLLM-MetaX source upgrade in ordered framework, model and patch phases, then run final model e2e and batched validation with a resumable full-file inventory. Use a domain skill for standalone component work.
---

# vLLM-MetaX Full Upgrade

First apply [environment-check](../environment-check/SKILL.md): approve the preparation
plan once and successfully install/verify both frameworks before any adaptation. An
installation failure blocks adaptation; source repairs cannot bypass this prerequisite.
Then read and apply [common-sense](../common-sense/SKILL.md). It owns
upstream-first source style, dependency closure, file ownership, evidence and recovery.
This coordinator owns the full inventory, phase order and whole-upgrade acceptance.

## Prepare the complete upgrade

Resume the existing full run, or initialize one before adaptation. Inventory every
`vllm_metax/` Python file and production/build/requirements file before edits. Assign
phase, owning skill, exact upstream/prior mapping and required acceptance. Keep added,
deleted and no-edit files in the ledger. Freeze the confirmed target revision and the
current-chip support matrix; inspect all target implementation paths, including branches
unavailable for runtime tests on the current hardware.

Assign [csrc-upgrade](../csrc-upgrade/SKILL.md) every native source/header/build contract,
regardless of whether production uses mcoplib. Place required native dependencies with
their earliest consumer and finish the remaining native sweep in framework.

Before entering platform, record `analysis-put` with the global dependency graph and
read-only contracts for every baseline model. Resolve cycles/ownership in the plan;
this planning step does not authorize early model construction or adaptation.

## Enforce the phase order

| Phase | Required source work |
| --- | --- |
| 1. `platform` | Review `platform.py` against target CUDA/base platform APIs and MACA contracts. |
| 2. `registry` | Upgrade all registrations, CustomOps, quantization configs and their local transitive kernels/experts. |
| 3. `attention` | Upgrade attention, quantized KV/indexer cache contracts and split prefill/decode execution. |
| 4. `framework` | Scan every remaining non-model/non-patch Python file, initializers and build/runtime dependencies, including a full csrc upgrade under either kernel provider. |
| 5. `models` | Upgrade every local model and model dependency; maintain each model's spec and prohibit dependency on upstream concrete model implementations. |
| 6. `patch` | Review every patch with removal as the objective; remove, replace via supported extension points, or minimize and justify. |
| 7. `validation` | Only now construct/load models and run model smoke, generation, e2e and batched inference correctness. |

Read [orchestration.md](references/orchestration.md) for domain boundaries and handoffs.
Use `phase-next` only after every file through the current phase has current source
adaptation and required local evidence. All four framework phases must finish before
models begin. Domain completion or one working model never substitutes for the full
framework upgrade. Do not schedule an early minimal model vertical slice.

Framework/operation unit tests and component numerical checks may run while adapting.
A model-free plugin/dispatch closure is allowed; complete-model construction, real
weight loading, dummy/trimmed smoke and generation all wait until phase 7.

## Handle discoveries without hiding unfinished work

The user permits required framework repairs discovered during models. Register a
separate earlier-phase dependency task with its owner and reason; invalidate affected
source/evidence, complete the repair and revisit consumers. The full framework gate
must be current before continuing model work or final model validation.

The user also permits an early **blocking** patch repair. Record its specific blocker
with `--exception-note`, scope it to resolving that blocker, and keep the comprehensive
patch review in phase 6. Early repair does not exempt it from the removal-first review;
record a fresh patch-phase review before counting its source adaptation complete.

## Final model validation and acceptance

Before any model execution, require current source readiness for the entire inventory,
including every model spec/import closure and all patches. Execute normal-entry model
e2e and `tools/batched_test` inference correctness for every supported matrix row and
at least every sample in `tools/models/`. Bind samples with `model-bind`. GPU-count/VRAM
shortages may be explicitly skipped using `model-skip`; all other missing coverage blocks
completion. Report skipped samples and required/available resources separately from passes.
Performance runs are additional only when requested; a benchmark alone is not correctness
acceptance. Read [common validation](../common-sense/references/validation.md) for actual
child/worker origins, numerical contracts and evidence limits.

Run `pre-commit run --all-files` after source adaptation and again on the final code
before completion. Fix every current hook failure, including pre-existing formatting
issues, and repeat until it passes. Review auto-fixes, reconcile source changes and
rerun affected required validation; preserve the final successful log with the run.
The [common validation procedure](../common-sense/references/validation.md) defines this
mandatory repository-wide acceptance check.

Run `finish` only when every inventoried file is marked **适配完成并验证完毕**, mappings and
all required evidence are current, every supported row has required e2e/batched
results, and the final full pre-commit run passes. Preserve explicit rejection evidence for unsupported rows. Record unavailable
chip quantization branches as unverified; report completion only within the agreed
current-chip scope, never as support for those branches. Required failures or blockers
within that scope prevent completion.

At handoff provide phase, source-ready and final-verified file counts, owner/blocker,
run/TODO/FILES paths, remaining matrix rows and the next executable action. A partial
framework or model subset cannot satisfy full completion.

For baseline samples using dummy weights, acceptance is explicitly `dummy_smoke`:
startup, valid prefill/decode generation and concurrent request/batch stability, without
semantic accuracy claims. Real-weight configurations use `accuracy` and their semantic
cases. Declare `configuration.validation_mode` in model rows and `load_format: dummy`
where applicable. A dummy smoke pass cannot satisfy real-weight accuracy acceptance.
