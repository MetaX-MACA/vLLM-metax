---
name: model-adaptation
description: Add or adapt complete independent MetaX model implementations using upstream-first ports, per-model difference specs, local model-private dependencies and concrete-model import audits. Also supplies shared model rules for model-upgrade.
---

# Model Adaptation

Apply [common-sense](../common-sense/SKILL.md) for the selected environment, exact
upstream-first code/style, dependencies, evidence and recovery. A standalone new model
uses a component run; a model in a full upgrade follows that run's phase gates. This
skill does not start a framework-wide upgrade merely because a new model is requested.

## Keep the complete model implementation local

Cover every requested registered model, backbone, draft/MTP/DSpark, multimodal branch
and model-private attention, indexer, compressor, cache, projection, router and kernels.
In a full run this is the models phase, after framework source adaptation is complete.

Copy the corresponding exact target model implementation and its model-related
implementation dependencies into `vllm_metax`, then reapply required MACA differences.
Do not import or inherit upstream **concrete model implementations**, whether they
live in `vllm.models.<model>`, a legacy concrete module under
`vllm.model_executor.models`, or are reached through a local helper, alias, lazy export
or dynamic import. Copy the model and its model-private dependencies into MetaX and
rewrite their imports locally. A local wrapper inheriting an upstream model does not
provide an independent implementation. This prevents upstream same-name model modules
and registration side effects from bypassing the plugin's own registered models.

Public framework APIs and shared utilities/interfaces remain upstream. For example,
`vllm.model_executor.models.utils`, `interfaces` and the common model registry are
permitted when used as shared APIs; the word `models` in a module path alone is not a
violation. Classify unfamiliar modules by their actual implementation and side effects.

For a model whose implementation lives under `vllm.models.<model>`, mirror the entire
model-owned directory into `vllm_metax.models.<model>`. Do not deduplicate non-vendor
files by hash or AST. Copy the model entry, `common/`, `nvidia/` and model-private
configuration/processors. Exclude non-NVIDIA vendor/platform implementations, including
`amd/`, `xpu/`, `cpu/` and equivalent directories. Rename the copied `nvidia/` directory
to `metax/`, then rewrite all imports, lazy references, registry strings and dynamic
module paths from the upstream `nvidia` prefix to the local `metax` prefix. Apply every
MACA adaptation in that renamed implementation.

The local mirror is the default ownership unit for a registered model:

1. `vllm/models/<model>/` maps wholly to `vllm_metax/models/<model>/`.
2. `vllm/models/<model>/nvidia/` maps wholly to
   `vllm_metax/models/<model>/metax/`.
3. Non-NVIDIA vendor/platform directories are excluded from the mirror; do not keep
   local copies of `amd/`, `xpu/`, `cpu/` or equivalent platform-specific branches.
4. Rewrite imports inside the mirror to local `vllm_metax` modules.
5. Keep framework APIs outside the model directory upstream when they are shared
   utilities/interfaces rather than concrete model implementations.
6. A local registration must point to the local mirrored model entry, and the actual
   worker must load the local implementation.

An unchanged non-vendor file inside the mirrored model directory is still retained. Its
lack of a MACA diff is recorded as a deliberate no-difference decision, not grounds for
deletion. An existing separate local `nvidia/` tree or a minimal wrapper-only adaptation
should be migrated into the complete mirrored `metax/` tree. Existing `amd/`, `xpu/`,
`cpu/` and equivalent local vendor paths should be removed once their consumers are
rewritten to the NVIDIA-derived local implementation.

The import auditor must not treat `vllm.models.<model>` as a permitted boundary for a
model that has a local mirror. It may permit only explicitly reviewed shared framework
APIs outside the model family. Thus `vllm.models.qwen4_exp.common...`,
`vllm.models.deepseek_v4...`, and similar same-family imports from the corresponding
local model are prohibited.

Run [audit_imports.py](scripts/audit_imports.py) on model entries and their reachable
local helpers without importing the plugin. It reports concrete implementation imports,
nonliteral dynamic imports, source hashes and public API boundaries. A static pass is
not a proof about every dynamic/runtime dispatch: review unresolved lookups and confirm
actual local model classes in final workers. Shared upstream framework internals are
outside the prohibition; delegating actual model construction to an upstream concrete
model is prohibited even through a public-looking factory.

## Maintain a spec for every model

Before model adaptation, create or update the model's own spec in its implementation
directory. Use `SPEC.md` for a dedicated model directory, or `<model>.SPEC.md` beside a
flat model file; give separately maintained models their own spec. The run links the
spec but is not its canonical location. Follow [spec-guide.md](references/spec-guide.md).
Keep it current with the final implementation and verified/unverified scope.

For every model record exact target/prior identities, the CUDA-to-MACA difference
matrix, the complete local mirror contents, `nvidia/ -> metax/` import mapping, local
import/dependency ownership, weight mapping and post-load transforms, execution paths,
capability limits and acceptance evidence. Do not leave a spec as an unverified list of
upstream defaults.

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

In final validation prove actual model/worker dispatch uses local implementations and
has no dependency on an upstream concrete model implementation. Exercise loading in the declared format (real checkpoint loading for real weights), post-load parameter state,
prefill/decode and supported modality, LoRA/MTP, graph and parallel branches. Link the
normal-entry e2e and batched inference results to the model spec and configured matrix.
Warnings, mocks or a successful local class import cannot satisfy model isolation.

For a bug also present upstream, retain the smallest justified model-local workaround
with concise scope/removal notes. Changes to other components for that shared bug need
separate authorization unless they are required dependency updates within this upgrade.
Keep capability errors intact; a warning cannot enable an unsupported path.

For baseline samples using dummy weights, acceptance is explicitly `dummy_smoke`:
startup, valid prefill/decode generation and concurrent request/batch stability, without
semantic accuracy claims. Real-weight configurations use `accuracy` and their semantic
cases. Declare `configuration.validation_mode` in model rows and `load_format: dummy`
where applicable. A dummy smoke pass cannot satisfy real-weight accuracy acceptance.
