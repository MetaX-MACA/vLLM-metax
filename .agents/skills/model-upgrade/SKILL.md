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
every required model variant and dependency. Copy the full target implementation, then
reapply justified MACA differences using common's preservation rules. Update the model's
canonical spec beside its code and rerun the concrete-import audit.

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
