---
name: vllm-metax-registry-upgrade
description: Review and adapt vllm_metax/registry registrations, quantization configurations, CustomOps and kernel dispatch against a target vLLM revision and installed MetaX APIs. Verify registration, inherited state, weight layouts and actual backend selection. Use for registry compatibility work, not standalone attention algorithms or monkey patches.
---

# vLLM-MetaX Registry Upgrade

Maintain the full contract from plugin activation to the selected implementation.
A registered class name, matching signature, or upstream support declaration does
not establish that the installed MetaX implementation supports the same behavior.

## Scope and environment

- Apply to requested adaptations in `vllm_metax/registry/`, including quantization
  configurations, CustomOps and linear kernels. Inspect directly related oracles,
  experts, wrappers and callers outside that directory as necessary. Edit them only
  when needed for the authorized adaptation; do not expand into unrelated changes.
- For sparse indexers registered here, cover registration, constructor compatibility,
  dtype dispatch and wrapper contracts. Standalone attention algorithm, cache-index
  or numerical kernel work belongs to the attention workflow. Monkey patches belong
  to the patch workflow. Do not automatically invoke either skill, impose patch
  headers, or write registry findings into `vllm_metax/patch/AUDIT.md`.
- Keep review-only requests read-only. Identify the subject: staged snapshot, working
  tree, installed package or specified revision. Preserve user changes and do not
  stage, commit, reinstall packages or edit upstream/site-packages without authorization.
  For staged validation, export the index and prove that imports use that snapshot;
  unstaged files must not silently supply missing staged dependencies.
- Read applicable repository instructions.
- Read and apply [vllm-metax-upgrade-common](../vllm-metax-upgrade-common/SKILL.md)
  before compatibility decisions. It owns environment/source discovery, the shared
  read-only probe, one-time target confirmation and verification evidence rules.
  Reuse the same established environment record across upgrade skills; do not ask
  again for an unchanged mapping. Keep the domain-specific workflow below.

## Trace registration and dispatch

Inventory every requested registration, including inactive/conditional entries and
aggregate import files. Record the registry key or target, activation path, effective
class/callable, upstream counterpart, MetaX difference, decision and evidence.

Trace the actual chain:

```text
plugin entry point -> package imports -> decorator/table mutation
-> config lookup or CustomOp/platform dispatch -> selected method
-> backend oracle -> experts/prepare-finalize -> wrapper -> installed kernel
```

- Exercise fresh-process plugin startup as well as direct module imports. One missing
  symbol in an eagerly imported quantization module can break every registration.
  A partially populated registry after an exception is not successful startup.
- Resolve aliases, lazy exports, module redirects, duplicate keys and import ordering.
  Verify the effective lookup result and caller binding, not only decorator presence.
- Check exact platform keys (including OOT), dtype/capability predicates and explicit
  backend overrides. Preserve other platforms' entries and explicit user choices.
- For overridden classes, inspect the current MRO and inherited methods. A copied
  constructor's `super()` may now call a parent with a different signature. Bypassing
  that parent requires initializing all state consumed by inherited allocation,
  loading, quant-config and execution methods.
- Check actual factory signatures and return contracts. Follow the selected class's
  module/source through kernel invocation. A MetaX-named oracle can still return
  upstream experts and bypass MCOPLIB, local tuning or communication adaptations, which
  could lead to potential errors.
- Distinguish shared and separately defined enums. Matching member names or values
  do not make members of different Enum classes interchangeable in backend dispatch.

## Verify semantics and adapt

For quantization work, read [quantization-contracts.md](references/quantization-contracts.md).
For CustomOps and linear registrations, apply the same chain checks to real/fake schemas,
optional inputs, tensor/tuple outputs, layout requirements and capability predicates.

Choose per entry: retain a needed difference, update the adapter, remove redundant code,
migrate to a supported extension point, or reject an unsupported combination early.
Do not copy upstream support claims into MetaX predicates without checking the installed
implementation. Comments and earlier review conclusions are hypotheses to verify.

- Validate joint configurations: quantization format, symmetry, group size, activation,
  bias, device, parallel mode and selected backend. Supporting one dimension does not
  establish support for every combination.
- Keep allocation, checkpoint loading, post-load conversion, parameter replacement,
  quant-config construction and kernel consumption consistent. Returned converted
  tensors must actually reach the kernel through the layer/config used by execution.
- Keep MetaX-specific behavior only where justified. Removing a redundant override is
  valid when the inherited implementation and its dependencies provide the same behavior.
- Preserve algorithm and layout explanations. State dimension meanings, packing axes,
  stride requirements, conversion order and why the MetaX path differs. Do not impose
  monkey-patch headers on registry files.
- Attribute defects carefully: new adapter regression, existing adapter defect, target
  upstream defect, installed component bug or environment mismatch. To claim an upstream
  defect, inspect and reproduce the actual target path; limit the claim to that revision.

## Validate and report

Read [validation.md](references/validation.md) and select checks that discriminate the
changed behavior, using the common skill's isolated-validation and evidence rules.

Start with startup/lookup and interface checks, then validate affected state/layout and
real GPU behavior. An isolated constructor or scatter test may bypass an earlier failure
inside that test process, but record the bypass and keep the real function under review.
Such tests do not establish production reachability or kernel support.

For a re-review, account for every prior finding as fixed, partially fixed, still present,
or masked by an earlier failure/capability rejection. Do not stop after fixing the first
exception. A rejected asymmetric configuration can hide a broken zero-point scatter.

Report locations, trigger, consequence, evidence, verification level and untested scope.
For fixes, include what changed, actual interpreter/import origins and reproducible test
commands/results. Record runtime workarounds explicitly. Do not claim full-model,
distributed, capture or performance validation from import and tensor-layout checks.
Use a user-requested audit location if provided; no separate audit file is mandatory.
