---
name: vllm-metax-attention-upgrade
description: Review and adapt MetaX attention backends, MLA, sparse indexers, cache layouts, and their kernel wrappers against a target vLLM revision and the actually installed MetaX component APIs. Verify dispatch, supported configurations, and GPU numerical behavior. Use for attention adaptations outside vllm_metax/patch/; do not apply monkey-patch headers or patch audit requirements.
---

# vLLM-MetaX Attention Upgrade

Preserve attention semantics while adapting to two independent contracts: the target
vLLM Python implementation and the installed MetaX kernel stack. Similar names,
matching signatures, successful imports, comments, and upstream tests do not establish
MetaX compatibility. Validate the actual runtime implementation.

## Scope and baseline

- Cover requested attention backends under `vllm_metax/v1/attention/`, common MLA
  layers, sparse indexer CustomOps, and directly related models and wrappers.
  Follow callers outside these paths when needed; do not expand into unrelated changes.
- Keep review-only requests read-only. For fixes, preserve staged and unstaged user
  changes. Do not stage, commit, reinstall dependencies, edit site-packages, or modify
  the upstream checkout unless the task authorizes that action.
- This is independent of the patch-upgrade skill. Do not apply patch-specific headers
  or write attention findings into `vllm_metax/patch/AUDIT.md`. If a task also changes
  monkey patches, use the patch workflow only for that portion.
- Read applicable repository instructions. Identify whether the requested subject is
  the index, working tree, installed release, or another revision. For staged reviews,
  export the index to a temporary directory and test that snapshot; unstaged fixes
  must not silently satisfy its missing dependencies.
- Read and apply [vllm-metax-upgrade-common](../vllm-metax-upgrade-common/SKILL.md)
  before compatibility decisions. It owns environment/source discovery, the shared
  read-only probe, one-time target confirmation and verification evidence rules.
  Reuse the same established environment record across upgrade skills; do not ask
  again for an unchanged mapping. Keep the domain-specific workflow below.
- Inventory every requested file: active registration/callers, upstream counterpart,
  reason for each MetaX difference, proposed action and validation. Account for retained
  and inactive adaptations as well as changed ones; do not let a passing subset stand
  in for the requested review scope.

## Trace the real component contracts

Read [component-contracts.md](references/component-contracts.md) for every affected
component family before deciding compatibility. Its versioned observations are prompts
for revalidation, not unconditional capability rules.

Trace each path from registration/platform selection through metadata builder, layer
binding, wrapper, imported callable, and installed Python/native kernel. Include native,
flattened, fallback, prefill/decode, and relevant capture/distributed branches.

For each used component entry point, record:

| Evidence | What to establish |
| --- | --- |
| Runtime identity | Distribution version, imported file, actual callable module/source, native extension path, active plugin and relevant feature flags. |
| Inputs | Positional/keyword arguments, tensor vs tuple, dtype, shape, strides, contiguity, scales, index units, and meanings of omitted/None arguments. |
| Outputs | Tensor vs tuple, output and LSE shapes/dtypes, LSE logarithm base, padding, aliasing, in-place writes and empty-row behavior. |
| Supported combinations | Joint constraints on dtype, QK/V dimensions, sinks, block size, quantization, native MTP, DCP, variable lengths, and graph capture. |
| Evidence level | Source-inspected, signature-checked, runtime-reproduced, numerically validated, or untested. |

Record the GPU model, available devices, Torch/Triton build identities and MACA
runtime/ABI evidence with runtime results. NVIDIA architecture checks in upstream
code are not substitutes for the executing MetaX device's capabilities.

Inspect `inspect.signature`, `inspect.getsourcefile`, extension docstrings and wrapper
bodies where available. Trace `*args/**kwargs`, decorators, lazy symbol resolution, and
aliases to the final implementation. An absent Python signature does not imply missing
support. An accepted keyword does not prove its semantics are implemented.

Actively challenge comments such as "FA2 cannot do DiffKV", "all sinks supported",
"FP8/FP4 share this API", "cache is contiguous", and "DCP supported". Compare claims
with executable restrictions and discriminating calls. Do not rewrite platform code
solely to resemble upstream, nor remove a fallback solely because upstream added a feature.

## Review attention semantics end to end

- Follow allocation/spec/layout selection through `bind_kv_cache`, cache insertion,
  gather, index conversion, kernel reads, and output merge. A logical shape does not
  prove a physical layout. Check layer/page/head/token strides and storage offsets.
- Preserve actual query counts, per-request boundaries and phase when reordering.
  Review uniform MTP, variable query lengths, short prefills, zero-length padding and
  CPU/device boundary differences. A reshape requires proven uniformity; otherwise
  flatten or bucket using correct per-token metadata.
- Distinguish request-relative token indices, paged slot indices, physical flat rows,
  and chunk-workspace offsets. Preserve original indices if consumers need different
  mappings. Review valid-count semantics when `-1` entries have interior holes.
- Treat dense-MHA-prefill tokens excluded from the MQA query as a valid subset. Do not
  run prefill work for absent tokens or return padded heads as real heads.
- For FP8/INT8, verify storage layout, scale placement/type, query scaling and whether
  weights already incorporate a scale. Do not infer semantics from dtype names.
- Check LSE layout/base, neutral empty outputs and distributed ownership before merging.
  DCP per-token causal lengths must be localized in the correct order relative to MTP
  expansion. Derive kernel head counts from gathered queries, not just local TP heads.
- Trace graph metadata lifetime and lazy scheduler state. Captured pointers must refer
  to runtime-updated buffers. Warmup success is not proof that changed inputs replay
  correctly. Metadata reuse must respect each installed component's conditions.

## Make a complete, bounded adaptation

Choose between retaining a necessary MetaX difference, updating an interface/semantic
mapping, using a verified fallback, or rejecting an unsupported configuration early.
For a suspected component defect, reproduce it by calling that component directly in
an isolated process before attributing it to the adapter.

- Update capability declarations and dispatch together: `supports_combination`, dtype
  and layout restrictions, `supports_out`, graph support, and constructors must agree
  with reachable implementations. Check combinations, not isolated flags.
- Keep verified fast paths. A local fallback should target the failing case, preserve
  masks/dtypes/outputs, avoid full-cache copies and hot-path host synchronization, and
  state why it exists and what evidence would permit removal. Do not promise speedups.
- Preserve useful algorithm comments. Explain the exact MetaX/upstream difference,
  affected installed component evidence, index units and removal/revalidation condition.
  Do not impose monkey-patch headers on these standalone implementations.
- Treat optional types and metadata contracts as part of compatibility. Use direct
  `is not None` narrowing where needed rather than hiding type errors with suppressions.

## Validate and report

Use the affected cases from [validation.md](references/validation.md). Run actual kernels
against independent numerical references in the intended environment. Test wrappers and
real branch routing in addition to raw APIs. A mocked conversion, isolated metadata test,
or successful import is not a full kernel/end-to-end test.

For now, do not depend on the repository's `tests/` directory, including its helpers,
fixtures, conftest files or existing test cases. Generate self-contained validation
scripts or tests as needed from the current runtime contracts and affected behavior.
Place them in an isolated temporary directory and provide their own inputs, references
and initialization. Keep the actual component under test real, not mocked.

For each finding or fix, retain the trigger, expected/observed behavior, component and
source origins, reproduction command, result, and verification limits. Distinguish a
new adaptation regression, an existing adapter defect, and an installed component bug.
Keep generated runtime scripts/tests in a clearly identified reproducible artifact;
do not depend on old `/tmp` files being present in future sessions.

Run relevant formatting, lint, type and diff checks. If a checker is unavailable in the
selected environment, say so rather than claiming it passed. End with concrete changes
or review findings, evidence, and untested scope. Place any requested audit alongside the
attention work or at a user-selected path, never in the patch audit by default.
