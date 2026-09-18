# Patch Review Patterns

These examples come from a vLLM-MetaX upgrade. Upstream versions change: inspect the
actual target implementation before using an example to decide whether a patch is needed.

## Large function copies and new extension points

A patch intended only to propagate HF overrides to a draft configuration may replace
all of `SpeculativeConfig.__post_init__`. That also freezes model detection, Medusa
handling, weight-source selection, and validation.

Look for a composition hook such as `compose_draft_hf_overrides`. If upstream already
supports callable overrides but lacks the dictionary inheritance MetaX requires, add
only that behavior and delegate the rest. Validate nested configuration types, override
semantics, isolation across repeated calls, and pickling. Do not assume every override
should propagate to every draft model.

## Quantization arguments can select execution modes

The upstream `per_act_token=True` branch of `_int8_quantize` may call
`per_token_quant_int8(A)`, ignoring the supplied scale. If the MetaX replacement
`scaled_int8_quant` selects static per-tensor quantization for a non-null scale,
forwarding `A_scale` unconditionally changes the branch's semantics.

Read the actual wrapper and kernel dispatch. Explain the affected branch, whether
behavior is unchanged for `None`, and what happens for a supplied value. Distinguish
the input scale from newly returned scales; static branches still need their scale.
Do not generalize this into a rule that quantization must never receive a scale.

## Tokenizer behavior follows the serialized backend

The same Llama tokenizer class can wrap ByteLevel or Metaspace/ByteFallback behavior.
Forcing every decoder to ByteLevel broadens the patch beyond the affected checkpoints.

Trace `from_pretrained -> convert_to_native_format -> __init__`. Determine whether
conversion extracts only vocabulary and merges while discarding the serialized decoder
or pre-tokenizer. Where a complete backend-loading path exists, preserve the file's
behavior and keep native initialization when no serialized file is available. Test
spaces, newlines, and another tokenizer family, not just successful initialization.

If borrowing an internal flag to select a local loading branch, inspect its execution
scope and downstream propagation. Do not infer safety from the flag's name or enable
remote-code permissions at the outer loading API.

## Batch regions and ordering within each region

One upstream revision expects `decode -> short_extend -> long_extend -> prefill` and
lets backends decide whether short extends count as decodes. Classification depends on
computed tokens, total prompt length, and tokens scheduled this step. A short query
does not establish that prompt processing is complete.

MetaX can preserve the four-region contract while stably sorting true decodes by query
length to produce contiguous buckets. Document:

- Whether upstream guarantees region placement but not decode query-length ordering.
- Whether a stable sort covers the entire batch or only misplaced requests. Stability
  within a subset does not guarantee stability across the whole region.
- Whether the early return checks region labels or the complete target permutation.
- Whether swaps only move misplaced requests or maintain both original indices and
  current-position mappings.
- Activation conditions, backend splitting behavior, and CPU sorting/swap overhead.
  Do not promise a speedup without measurement.

Keep the original algorithm explanation and update its examples. With threshold 3,
`[D:3, D:1, D:2, S:2, L:8, P:4]` already satisfies region placement, while MetaX may
produce `[D:1, D:2, D:3, S:2, L:8, P:4]`. Define D/S/L/P and clarify that the numbers
represent tokens scheduled this step.

## Registries and explicit configuration

If a kernel type lacks a public registration API, temporarily add or update the OOT
entry instead of replacing the dictionary and deleting other platforms' entries.
Check whether candidate order determines priority and whether registration can repeat.

When synchronizing device visibility, distinguish a missing source variable, an explicitly
empty source value, and an explicitly configured destination value. Do not overwrite
settings with an unconditional empty-string default. Avoid mutating the original RPC
input when constructing a worker environment.

## Redundant replacements and cleanup

An allocator patch may already be covered by a module redirect in the plugin entry point.
Before removing it, trace import timing, platform checks, and caller bindings to establish
that upstream will actually resolve to the MetaX allocator.

Do not retain an old `shutdown` copy merely because it appears functional. Upstream may
have added communication, elastic executor, model runner, or memory-pool cleanup. Verify
that removing the copy preserves both MetaX allocator dispatch and upstream cleanup.

## Identical kernel bodies do not establish redundancy

A Triton kernel body can match upstream while autotune warp/stage counts, minimum MMA
tiles, or shared-memory requirements remain unsuitable for MetaX. Compare decorators,
heuristics, launch arguments, and callers as well as the function body.

When upstream implements the original fix, such as vocabulary tiling, also compare tail
masks, bounds checks, and tie-breaking. The old patch may overwrite a more complete
upstream fix. Both retention and removal require evidence from the target version.
