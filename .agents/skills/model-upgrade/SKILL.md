---
name: model-upgrade
description: Upgrade existing MetaX models across a confirmed upstream revision range using the independent model implementation, spec and import-audit rules from model-adaptation.
---

# Model Upgrade

Apply [common-sense](../common-sense/SKILL.md) and
[model-adaptation](../model-adaptation/SKILL.md). Model independence, complete local
model-private implementations, concrete-import boundaries, per-model specs and model
acceptance are defined there and are not duplicated in this skill.

In a full run, wait for the framework source gate before editing models. Before the
framework phases, perform read-only analysis of every model's required framework
interfaces, cache layouts and quantization contracts in the shared dependency plan.

For each existing model, compare its prior upstream base and exact target, inspect
relevant upstream commits and moved/split/target-only implementations, and account for
every required model variant and dependency. Use the full target implementation as the
review baseline. Mirror the complete model-owned upstream directory locally and adapt
its NVIDIA implementation as MetaX code. Update the model's canonical spec beside its
code and rerun the concrete-import audit.

## Mirror the complete model directory

When `vllm.models.<model>` is the implementation owned by a registered MetaX model:

- Copy the entire upstream `vllm/models/<model>/` directory to
  `vllm_metax/models/<model>/`, including the model entry, common helpers, processors
  and configuration code that belong to that model directory.
- Exclude every non-NVIDIA vendor/platform implementation from the copy, including
  `amd/`, `xpu/`, `cpu/` and equivalent platform-specific directories. MACA derives
  from the NVIDIA implementation; do not retain other vendors' local implementations.
- Rename the copied `nvidia/` directory to `metax/`. Rewrite every import, lazy
  reference, registry path and dynamic module string from
  `vllm.models.<model>.nvidia` to `vllm_metax.models.<model>.metax`.
- Treat the renamed `metax/` directory as the place where all MACA adaptations are
  applied, using the NVIDIA implementation as the behavioral baseline.
- Do not retain a local `nvidia/` directory after migration. Existing MetaX changes
  must be reapplied inside the new `metax/` implementation.
- Keep the remaining model-owned, platform-neutral directories and files from the
  upstream copy. Do not remove a non-vendor file merely because it is byte-identical,
  differs only by import paths, or has no current MACA modification. The complete
  NVIDIA-derived model directory is deliberately owned by the local model.
- Rewrite imports within the copied model directory so they resolve to that local copy.
  Do not mix local model modules with their upstream same-family counterparts.
- Shared framework APIs outside this model directory may remain upstream, such as
  `vllm.model_executor.models.utils`, `interfaces`, registry/config APIs and
  `vllm.models.common`. Dependencies that are model-owned concrete implementations must
  still be copied and localized according to the model spec.

The import audit must reject imports from `vllm.models.<model>` inside a locally mirrored
model. Upstream concrete model construction or inheritance is prohibited. The only
expected upstream imports from a mirrored model are explicitly reviewed framework/public
dependencies outside the copied model family.

If the upstream model directory has no `nvidia/` directory, retain the non-vendor model
files and apply MACA changes in the appropriate local model-owned modules; document why
there is no NVIDIA branch to rename. If an older local `metax/` tree contains adaptations
that predate the mirror, migrate those differences into the mirrored tree instead of
keeping two competing implementations.

Record in the model spec:

- the exact upstream directory and target revision copied,
- the local mirror path,
- the `nvidia/ -> metax/` mapping and rewritten import prefixes,
- all removed non-NVIDIA vendor/platform directories,
- every actual MACA difference applied in `metax/` or another model-owned file.

Validate the mirrored graph in both directions: the import audit must show that the
registered model resolves through the complete local mirror, and final model/worker
execution must show the local architecture and `metax` implementation. Normal-entry e2e
and batched validation remain mandatory.

The minimum final validation set includes every model sample currently discovered under
`tools/models/`; multiple samples of the same architecture are distinct configurations.
Bind each to its spec, source-review task and exact e2e/batched rows with `model-bind`.
Only measured GPU-count/VRAM shortages may use a recorded resource skip; missing weights,
implementation failures or absent test bindings must not silently count as skips.

For baseline samples using dummy weights, acceptance is explicitly `dummy_smoke`:
startup, valid prefill/decode generation and concurrent request/batch stability, without
semantic accuracy claims. Real-weight configurations use `accuracy` and their semantic
cases. Declare `configuration.validation_mode` in model rows and `load_format: dummy`
where applicable. A dummy smoke pass cannot satisfy real-weight accuracy acceptance.
