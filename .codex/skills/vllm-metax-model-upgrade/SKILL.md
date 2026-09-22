---
name: vllm-metax-model-upgrade
description: Review and upgrade MetaX model support against a target vLLM revision and installed MACA components, recursively including model-dependent attention and kernels. Establish quantization and cache differences, preserve upstream structure, and distinguish shared upstream bugs from adaptation defects. Use for model support work, not standalone monkey-patch or registry audits.
---

# vLLM-MetaX Model Upgrade

Preserve the target upstream structure while implementing the verified MACA execution
contract. A successful upgrade includes the model's reachable dependencies, not just
its top-level Python file. Upstream support declarations and comments are evidence to
investigate, not proof of support on MetaX.

## Scope and environment

- Cover the requested models under `vllm_metax/models/`, their registration and
  configuration, and recursively their required attention, cache, compressor, indexer,
  projection, MTP/draft, MoE and kernel wrappers. Keep unrelated components out of scope.
- Keep review-only requests read-only. Establish whether the subject is the index,
  worktree, installed package or specified revision. Preserve staged/unstaged changes;
  do not stage, commit, reinstall dependencies or edit upstream/site-packages unless
  authorized. For staged validation, export and import the index snapshot explicitly.
- Read applicable repository instructions.
- Read and apply [vllm-metax-upgrade-common](../vllm-metax-upgrade-common/SKILL.md)
  before compatibility decisions. It owns environment/source discovery, the shared
  read-only probe, one-time target confirmation and verification evidence rules.
  Reuse the same established environment record across upgrade skills; do not ask
  again for an unchanged mapping. Keep the domain-specific workflow below.
- Apart from the common prerequisite above, do not automatically invoke patch, attention or
  registry skills just because the model calls those modules. Apply a separately
  requested specialist workflow only to its relevant portion. No monkey-patch headers
  or patch audit files are required for model adaptations.

## First establish the MACA/upstream differences

Before deciding what to copy or remove, build an evidence-backed difference matrix
for each requested model family. Include the effective configuration after platform
and speculative-decoding rewrites, not only the original HF config.

| Contract | Record for both upstream and MACA |
| --- | --- |
| Weights | Checkpoint dtype, runtime dtype, per-channel/block/MX grouping, scale encoding, packing, post-load conversion and excluded modules. |
| Activations and projections | Compute/output/accumulation dtype, fused vs separate projections, grouped GEMM, padding, quantization and inverse RoPE order. |
| KV and indexer caches | Logical dtype vs physical storage dtype, quantization scales, compressed/SWA separation, page layout/stride and index units. |
| Attention | Selected backend, model-specific attention, prefill/decode/MTP paths, head dimensions, sinks, causal behavior, compression and capability restrictions. |
| MoE and parallelism | Actual experts implementation, routing/scaling, shared experts, TP/PP/SP/EP/DCP collectives and supported combinations. |
| Optional features | LoRA, MTP/DSpark, graph capture and each feature's real dispatch and restrictions. |
| Components | Imported DeepGEMM, FlashAttention/FlashMLA, MCOPLIB and other actual callables, signatures, extension identities and verified semantics. |

Separate intentional MetaX differences, inherited upstream behavior, obsolete overrides,
adaptation defects and unverified assumptions. State unknowns explicitly. Inspect actual
APIs, wrapper bodies and executable capability checks; challenge existing comments.
A dtype name, matching signature or NVIDIA architecture predicate is not a MACA contract.
Do not overwrite a necessary BF16/INT8 path with upstream FP8/FP4 behavior merely to
reduce the diff. Structural alignment must preserve the established MACA semantics.

## Recursively inventory and update dependencies

Trace from model registration through config rewriting, layer construction, weight
loading, execution dispatch and kernels. Maintain a visited dependency inventory with:
local symbol/file, target upstream counterpart, callers, relevant upstream delta,
MetaX difference, action and validation evidence.

For each changed upstream model interface or behavior:

1. Inspect direct dependencies and their upstream changes.
2. Follow their dependencies until reaching a verified stable interface or an installed
   component boundary; record that boundary and why no further local change is needed.
3. Update affected local dependencies and callers together. Include model-private
   attention (such as DeepSeek V4 attention/compressor/FlashMLA code) even when it lives
   outside the main model file or a generic attention directory.
4. Inspect transitive consumers of shared helpers before changing their contracts.
   For external binary components, verify the installed API instead of silently
   upgrading or modifying the component.

Account for unchanged, removed, conditional and inactive files. Report blocked paths
rather than claiming completion from successful imports of a subset. Read
[review-cases.md](references/review-cases.md) for relevant model-specific checks;
its examples are investigation prompts, not permanent support restrictions.

## Keep the implementation easy to diff

- Preserve upstream class/function boundaries, names, signatures, method order, control
  flow and file organization wherever MACA semantics allow it. Follow upstream moves
  when practical; record explicit mappings where platform directories differ.
- Avoid unrelated refactors, reformatting, renaming, helper extraction and broad
  defensive scaffolding. Prefer a small change at the corresponding upstream location
  over a new abstraction that obscures future upgrades.
- Every retained or newly introduced MetaX ad-hoc must have a nearby `NOTE(MetaX)`
  comment explaining the concrete upstream difference, why MACA needs it, and when
  it can be removed or revalidated. Preserve useful algorithm and layout explanations.
  Do not use a generic "MetaX modification" marker as the entire explanation.
- Compare both the local change diff and the complete local-vs-target diff. The latter
  must expose necessary platform differences instead of being dominated by structural
  churn. Remove obsolete workarounds only after verifying the replacement path.

Example note (adapt the content to verified evidence):

```python
# NOTE(MetaX): The target upstream path quantizes this projection to FP8.
# This MACA path uses BF16 because <verified component constraint>.
# Preserve <layout/scale invariant>; revalidate when <capability> is available.
```

## Investigate upstream before fixing a suspected bug

For every potential bug found during validation, first inspect the equivalent path in
**the target upstream revision**, including dispatch and relevant dependencies. Where
feasible run the same discriminating reproduction. Classify the cause as an adaptation
regression, existing MetaX defect, shared upstream defect, component issue or environment
mismatch. Distinguish source-based suspicion from reproduced upstream failure.

If upstream has or may have the same issue:

- First look for the smallest model-local fix or workaround that preserves upstream
  structure and the intended MACA behavior. When making such a fix, add a nearby
  `NOTE(MetaX)` explicitly saying the target upstream may also be affected, citing the
  symbol/revision and evidence or uncertainty. Explain the local workaround and its
  removal condition. Do not imply it is a MetaX-only adaptation defect.
- If resolving that shared defect requires changing other modules/components, do not
  expand the fix into those dependencies for this bug. Add a targeted `logger.warning`
  (or `logger.warning_once` when available) and report the limitation and repair plan.
  Prefer configuration/construction time over repeated token-time logging. Explain
  the trigger, consequence and known workaround without claiming the bug is fixed.
- Preserve existing rejection/capability checks. A warning does not establish support,
  justify enabling an unsupported path or replace an existing error with silent success.
- This warning-only boundary concerns repairs to shared upstream bugs. It does not
  cancel the recursive dependency updates required for the requested model upgrade.
  Follow explicit user authorization if they separately request the cross-module fix.

Example shared-defect note:

```python
# NOTE(MetaX): Target upstream <revision/symbol> also appears to bypass <contract>.
# <Evidence; state if source-inspected only>. Keep this workaround local by <action>.
# Revisit when upstream handles <condition>; do not remove on version alone.
```

## Validate and report

Generate focused, self-contained validation scripts in an identified temporary artifact
directory. For now, do not depend on repository `tests/`, helpers, fixtures or conftest,
or on old `/tmp` artifacts surviving from earlier sessions.

- Verify registration and fresh-process construction, then weight loading, transformed
  parameter state, dispatch and affected numerical behavior. Test meaningful boundary
  cases and unsupported combinations from the difference matrix.
- Use real installed kernels and independent numerical references where available.
  Test model wrappers as well as raw APIs. Isolated mocks can test constructor or routing
  contracts, but do not establish production reachability or GPU correctness.
- For parallel changes, check collective ownership and residual semantics. If multi-GPU
  execution is unavailable, distinguish constructor/source evidence from distributed
  numerical validation. Likewise, a wrapper check is not an end-to-end LoRA/MTP test.
- Record runtime origins and any process-local optional-dependency workaround. Run
  relevant formatting, lint, type and diff checks; disclose unavailable checks.
- Re-review every finding after edits, including prior failures masked by earlier
  exceptions. Check for concurrent source changes before reporting final results.

Report the target/environment, difference matrix, dependency coverage, concrete changes
or findings, upstream-bug attribution, warning-only unresolved cases, reproducible
validation and untested scope. No separate audit file is mandatory. Do not call a model
fully supported based only on imports, signatures or a passing kernel micro-test.
