---
name: model-adaptation
description: Add new MetaX models or upgrade existing ones to an exact vLLM revision. Maintain complete local model implementations, model specs and import audits in standalone work and full upgrades.
---

# Model Adaptation

Apply [common-sense](../common-sense/SKILL.md) for environment preparation, upstream-first
source adaptation, dependency ownership, evidence and recovery. Use a component run for
standalone model work and reuse the coordinator's run during a full upgrade.

## Establish the upstream baseline

- For a new model, confirm the exact target revision, checkpoint variants and upstream
  implementations to port. Mark the prior implementation as inapplicable for a first port.
- For an existing model, confirm its prior upstream base and exact target revision. Review
  the relevant upstream commits and account for moved, split, removed and target-only
  implementations, variants and dependencies. Start from the complete target implementation
  and reapply the MetaX differences that are still needed.

In a full upgrade, analyze each model's required framework interfaces, cache layouts and
quantization contracts in the shared dependency plan before framework edits. Begin model
edits after framework source adaptation is complete. Follow the
[coordinator's phase and validation gates](../vllm-metax-upgrade/SKILL.md).

## Keep the complete model implementation local

Cover every requested registered model, backbone, draft/MTP/DSpark, multimodal branch
and model-private attention, indexer, compressor, cache, projection, router and kernels.
Standalone upstream-owned models can keep using vLLM registrations when their target
configurations work on MACA. Apply the model ownership rules below to plugin-owned entries.

For an implementation under `vllm/models/<family>/`, mirror the full model-owned
directory into `vllm_metax/models/<family>/`. Keep the model entry, family-local
`common/`, configurations, processors and other platform-neutral files, including files
that need only import rewrites or have no MACA differences. Exclude non-NVIDIA vendor
implementations such as `amd/`, `xpu/` and `cpu/`. Map `nvidia/` to `metax/` when present,
using the NVIDIA implementation as the baseline for MACA changes. If the target has no
`nvidia/`, adapt the relevant local model-owned modules and record that layout in the spec.
Migrate prior local adaptations into the mirror and retire superseded vendor trees or
wrappers after migrating their consumers.

Point registrations at the local model entries. Rewrite intra-family imports, lazy
references, registry strings and dynamic module paths to the local mirror. For legacy
implementations under `vllm/model_executor/models/`, port the complete concrete model
implementation and localize its model-private dependencies. Apply this rule also to
concrete dependencies reached through a local helper, upstream alias, public-facing
factory, inheritance or dynamic dispatch.

### Shared framework dependencies

`vllm.models.<family>.common` belongs to that model family, so its consumers use the
local mirror. Cross-model APIs under `vllm.models.common`, and shared framework APIs
such as `vllm.model_executor.models.utils`, `interfaces` and the model registry, are
imported from upstream when their contracts work on MACA. Leave these shared modules out
of the model mirror. Trace the actual symbols, consumers and side effects to determine
ownership. If a shared API needs MACA adaptation, track the affected shared module or
extension point with its framework owner and recheck its consumers after the repair.

### Reuse an upstream model component

A plugin-owned model may reuse a component from a different upstream model when its
contract, transitive dependencies and actual MACA dispatch can be verified in the consuming
model. Keep model construction and registration local. Review the upstream component and
its side effects at the exact target commit. Scope each declaration to the local entry,
importing file, exact upstream symbol and its intended use. Record the rationale, applicable
configurations and runtime obligations in the consuming [model spec](references/spec-guide.md#upstream-component-reuse).

Run [audit_imports.py](scripts/audit_imports.py) from every registered model or variant's
local entry and include its reachable helpers. `model-review` checks exact declared symbol
imports against the selected upstream commit. Other concrete upstream model imports,
including same-family `common/`, remain prohibited. Review public API boundaries, reused
symbols and unresolved dynamic references in the audit artifact.

## Maintain a spec for every model

Before model adaptation, create or update the model's own spec in its implementation
directory. Use `SPEC.md` for a dedicated model directory, or `<model>.SPEC.md` beside a
flat model file; give separately maintained models their own spec. The run links the
spec but is not its canonical location. Follow [spec-guide.md](references/spec-guide.md).
Keep it current with the final MACA implementation so the next adaptation can reproduce it.
Keep run-specific audit data, status and validation artifacts outside the checkout.

For every model record exact target/prior identities, the CUDA-to-MACA difference
matrix, the complete local mirror contents and vendor mapping, local dependency
ownership and exact shared upstream imports, weight mapping and post-load transforms,
execution paths, capability limits and workarounds.

For DSv4 and other private-attention/indexer models, explicitly specify both upstream
and MetaX attention **and indexer** cache quantization, physical block byte layout,
strides, scale representation/placement, packing, padding/alignment, block/token index
units and the complete insert/compress/gather/read path. Include prefill/decode split,
mixed batches, compression/SWA, MTP/DSpark and inverse RoPE/grouped projection contracts.

## Establish model contracts

- Read the actual checkpoint variant's official config. Map backbone and draft/MTP
  layers separately; verify indexed arrays after platform/speculative rewrites.
- Distinguish checkpoint, runtime weight, activation, attention-cache and indexer-cache
  dtypes. Follow fused mappings, scales, sharding, repacking and loading order into the
  tensor consumed by the kernel. A checkpoint label does not establish MACA compute.
- Trace registration and rewritten config to the intended **local** decoder/model,
  actual parameters, shared experts, wrappers, projection and kernels.
- Check TP/PP/SP/EP/DCP collective ownership, residual state and sharding. Review LoRA
  execution rather than equating a rank field or interface with actual adapter support.
- Read [review-cases.md](references/review-cases.md) for relevant loader, private cache,
  model-identity and distributed failure patterns.

If a new framework dependency needs repair, register its own owner/task and invalidate
consumers. Such repairs are permitted during models; they do not permit early model
execution. A blocking patch repair needs the coordinator's recorded exception.

## Model-specific acceptance

During adaptation review the static import graph, aliases, dynamic import sites and
local model dependencies; document unresolved dynamic edges. Use isolated layer/cache
checks under common's rules without constructing/loading a complete model. Record
per-model spec and import-closure review as source acceptance obligations.

In final validation confirm the worker uses the local model and only its declared upstream
component dependencies. For each declared use, exercise the consuming model's actual branch
through its normal entry point. Assert the component origin and actual MACA backend in the
worker report for the relevant support rows. An upstream model pass supplies context for
review, while acceptance evidence comes from running the consuming model. Re-review and
validate the uses when the upstream commit or component contract changes.

Exercise loading in the declared format (real checkpoint loading for real weights), post-load parameter state,
prefill/decode and supported modality, LoRA/MTP, graph and parallel branches. Link the normal-entry e2e and batched inference results to the reviewed specs and
configured matrix in the external run.
Warnings, mocks or a successful local class import cannot satisfy model isolation.

For a bug also present upstream, retain the smallest justified model-local workaround
with concise scope/removal notes. Changes to other components for that shared bug need
separate authorization unless they are required dependency updates within this upgrade.
Keep capability errors intact; a warning cannot enable an unsupported path.

Declare `dummy_smoke` or `accuracy` for each configured model row and apply the
[shared acceptance and sample binding rules](../common-sense/SKILL.md#baseline-model-acceptance).
In a full upgrade, bind every baseline sample with `model-bind`. For standalone work,
validate the declared model and configuration scope.

At completion, write a review report outside the repository. Diff the final tree
against the pre-upgrade MetaX baseline. For every changed file record the reason,
intended behavior, implementation and relevant evidence or coverage limits.
Include additions, deletions, shared dependencies, build, tests and documentation.
