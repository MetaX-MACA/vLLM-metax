---
name: attention-upgrade
description: Review and adapt MetaX attention backends, MLA, sparse indexers, cache layouts, and kernel wrappers against target vLLM and installed MetaX component contracts. Use for attention work outside vllm_metax/patch/.
---

# vLLM-MetaX Attention Upgrade

Fully read and apply [common-sense](../common-sense/SKILL.md) before this workflow.

Cover the requested attention backends under `vllm_metax/v1/attention/`, common MLA
layers, sparse indexer CustomOps, and their shared wrappers. Model-private implementations belong to model-upgrade.

## Trace component contracts and dispatch

Read [component-contracts.md](references/component-contracts.md) for each affected
component family. Its versioned observations are investigation prompts, not permanent
capability rules.

Trace registration/platform selection through metadata builder, layer binding, wrapper,
and the selected Python/native kernel. Include native, flattened, fallback,
prefill/decode, capture and distributed branches relevant to the requested change.
Resolve aliases, decorators, `*args/**kwargs` and lazy symbols to the actual FA, MLA or
indexer entry point; wrapper transformations can differ from raw component contracts.

| Contract | Establish for each selected entry point |
| --- | --- |
| Inputs | Positional/keyword arguments, tensor vs tuple, dtype, shape, strides, contiguity, scales, index units and meanings of omitted/None arguments. |
| Outputs | Tensor vs tuple, output and LSE shapes/dtypes, LSE logarithm base, padding, aliasing, in-place writes and empty-row behavior. |
| Supported combinations | Joint restrictions on dtype, QK/V dimensions, sinks, block size, quantization, native MTP, DCP, variable lengths and graph capture. |

Treat every API change in target attention, MLA, indexer, PCP/DCP and DeepGEMM wrappers
as upstream-owned. Port the target signature and argument meaning first, including
parameters that carry metadata-routing or lifecycle state, then adapt installed MetaX
components inside the wrapper and migrate all local callers. Keep compatibility inside
the implementation boundary rather than changing the public API; if a target API cannot
be supported, record the exact component limitation and reject it explicitly.

Challenge claims such as "FA2 cannot do DiffKV", "all sinks supported", "FP8/FP4 share
this API", "cache is contiguous" and "DCP supported" with executable restrictions and
discriminating calls. An absent Python signature can be an extension-interface detail;
an accepted keyword can still have unimplemented semantics. A fallback remains needed
until the selected MetaX entry point implements its replacement.

## Review attention semantics end to end

- Follow allocation/spec/layout selection through `bind_kv_cache`, insertion, gather,
  index conversion, kernel reads and output merge. Check layer/page/head/token strides
  and storage offsets; logical shapes do not establish physical layout.
- Preserve actual query counts, per-request boundaries and phase when reordering.
  Review uniform MTP, variable query lengths, short prefills, zero-length padding and
  CPU/device boundary differences. Reshape only with proven uniformity; otherwise
  flatten or bucket with the correct per-token metadata.
- Distinguish request-relative token indices, paged slots, physical flat rows and chunk
  workspace offsets. Preserve original indices when consumers need different mappings.
  Review valid-count semantics when `-1` entries have interior holes.
- Dense-MHA-prefill tokens excluded from MQA form a valid subset. Do not run prefill
  work for absent tokens or return padded heads as real heads.
- For FP8/INT8, verify storage layout, scale placement/type, query scaling and whether
  weights already incorporate a scale.
- Check LSE layout/base, neutral empty outputs and distributed ownership before merging.
  Localize DCP causal lengths in the correct order relative to MTP expansion. Derive
  kernel head counts from gathered queries rather than local TP heads.
- Trace graph metadata lifetime and lazy scheduler state. Captured pointers must refer
  to runtime-updated buffers; metadata reuse must satisfy the component's conditions.
- Narrow optional attention metadata with direct `is not None` checks where required;
  do not hide incompatible metadata types with suppressions.

## Quantized cache and split forwards

Treat attention KV cache and indexer cache as separate contracts. For each supported
format compare upstream and MetaX logical/physical dtype, scale representation and
placement, block byte layout, padding/alignment, token/head/page strides and index units.
Follow allocation, insertion/compression, gather and kernel reads together. Review
INT8 cache support explicitly; an upstream FP8 implementation is not a compatible
substitute for the MACA path.

Review separate prefill and decode forwards and their dispatch/metadata boundaries,
not only a combined forward. Include mixed batches, short prefills, multi-token decode,
MTP, per-request offsets and reuse of cache/graph buffers. Preserve the target split
and required MetaX semantics without merging/refactoring upstream paths. Apply common's
chip policy when selecting numerical cases; unavailable formats remain source-reviewed
and explicitly unverified on the current chip.

## Align capability declarations and implementations

Update `supports_combination`, dtype/layout restrictions, `supports_out`, graph support
and constructors with the reachable implementations. Evaluate combinations rather than
independent flags. Retain a verified implementation, update its mapping, select a
verified fallback or reject an unsupported combination before launching the kernel.

Keep verified fast paths. Limit a numerical fallback to its failing case, preserve
masks/dtypes/outputs, and avoid full-cache copies or hot-path host synchronization.
Document the component condition that permits its removal without promising speedups.

## Attention acceptance cases

Read [validation.md](references/validation.md) for affected numerical, metadata-routing,
graph-replay and DCP acceptance cases.
