---
name: registry-upgrade
description: Review and adapt vllm_metax/registry registrations, quantization configurations, CustomOps and kernel dispatch against a target vLLM revision and installed MetaX APIs. Verify registration, inherited state, weight layouts and actual backend selection. Use for registry compatibility work, not standalone attention algorithms or monkey patches.
---

# vLLM-MetaX Registry Upgrade

Read [common-sense](../common-sense/SKILL.md) in full and
apply it before this specialist workflow.

## Scope

- Cover requested `vllm_metax/registry/` adaptations: quantization configurations,
  CustomOps and linear kernels, plus directly required oracles, experts, wrappers
  and callers. A compressed-tensors MoE adaptation includes the selected local modular
  kernels, experts, oracles and prepare/finalize code even outside `registry/`.
- For sparse indexers registered here, cover registration, constructor compatibility,
  dtype dispatch and wrapper contracts. Standalone attention algorithms, cache indices
  and kernel numerics belong to the attention workflow; monkey patches belong to the
  patch workflow.

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

- Verify normal plugin startup completes as well as each registration's activation.
  One missing symbol in an eager quantization import can break all registrations;
  a partially populated registry after an exception is not successful startup.
- Resolve aliases, lazy exports, module redirects, duplicate keys and import ordering.
  Verify the effective lookup result and caller binding, not only decorator presence.
- Check exact platform keys (including OOT), dtype/capability predicates and backend
  overrides without disturbing other platforms' registry entries.
- For overridden classes, inspect the current MRO and inherited methods. A copied
  constructor's `super()` may now call a parent with a different signature. Bypassing
  that parent requires initializing all state consumed by inherited allocation,
  loading, quant-config and execution methods.
- Check actual factory signatures and return contracts. Follow the selected class's
  module/source through kernel invocation. A MetaX-named oracle can still return
  upstream experts and bypass the approved kernel provider, local tuning or communication
  adaptations. Verify actual callable origins against the selected provider.
- Distinguish shared and separately defined enums. Matching member names or values
  do not make members of different Enum classes interchangeable in backend dispatch.

## Verify semantics and adapt

For quantization work, read [quantization-contracts.md](references/quantization-contracts.md).
For CustomOps and linear registrations, apply the same chain checks to real/fake schemas,
optional inputs, tensor/tuple outputs, layout requirements and capability predicates.

- Validate joint configurations: quantization format, symmetry, group size, activation,
  bias, device, parallel mode and selected backend. Supporting one dimension does not
  establish support for every combination.
- Keep allocation, checkpoint loading, post-load conversion, parameter replacement,
  quant-config construction and kernel consumption consistent. Returned converted
  tensors must actually reach the kernel through the layer/config used by execution.
- Before removing an overridden registration, confirm the inherited factory still
  selects the intended implementation and initializes its consumed state.
- Check dimension meanings, packing axes, strides and conversion order across registered
  allocation, post-load conversion and kernel consumption.

## Validate registration and state

Read [validation.md](references/validation.md) for registration, weight-layout and kernel
cases. Check startup/lookup before state construction, post-load conversion and actual
kernel selection. Recheck affected asymmetric branches after resolving capability
rejection: that rejection can hide a broken zero-point scatter.
