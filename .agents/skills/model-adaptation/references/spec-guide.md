# Per-model upstream and MACA difference spec

Every maintained model has its own spec in its model implementation directory. Keep
model specs beside code and link them from the upgrade task. A dedicated directory uses
`SPEC.md`; a flat model file uses `<model>.SPEC.md`. Identify each model and draft variant
covered; independently maintained models need their own document.

Use these sections, with concrete values or explicit unknown/unverified entries:

1. **Identity and scope:** registered architectures, checkpoint variant, target/prior
   upstream revision and files, MetaX implementation files, retained features.
2. **Difference matrix:** upstream CUDA contract, MACA contract, reason, exact local
   implementation and eventual removal condition. Include deliberate no-difference
   decisions where they determine a shared contract.
3. **Local model dependencies:** the complete `vllm.models.<model>` directory mirrored
   into `vllm_metax.models.<model>` after excluding non-NVIDIA vendor/platform
   implementations, the local `nvidia/ -> metax/` mapping, rewritten
   imports/lazy references/registry strings, removed vendor directories, public
   framework API boundaries, and direct/indirect/dynamic concrete-model import audit
   evidence. State that unchanged non-vendor files remain part of the local mirror and
   which shared upstream framework APIs are used from outside the model directory.
4. **Weights and compute:** checkpoint/runtime/activation dtypes, quantization groups,
   scale formats and orientation, fused mappings, TP sharding, loading order, post-load
   transforms, MoE experts and actual component/kernel dispatch.
5. **Attention and indexer cache:** separate upstream/MetaX rows for each cache type.
   Give dimensions, logical/physical dtype, byte/block layout, strides, scale location,
   packing, page padding/alignment, block size and index units. Show producer-to-consumer
   mapping: allocate → compress/quantize → insert → gather/index → attention/kernel read.
   State prefill/decode/split/mixed-batch, compressed/SWA and MTP/DSpark differences.
6. **Other execution contracts:** normalization, residuals, projections/inverse RoPE,
   routing/shared experts, TP/PP/SP/EP/DCP collectives, multimodal and adapter branches.
7. **Chip scope and validation:** available-chip configurations, source/local acceptance,
   final real-load/e2e/batched result paths, numerical cases and limits. Unavailable
   FP8/FP4 branches remain explicitly unverified, with the condition to validate later.
8. **Open obligations:** precise blocker, affected paths, owner and next action. Do not
   replace an unknown byte layout or unsupported claim with a guessed upstream value.

For DSv4 cache specs, include attention and indexer quantization independently,
compressed/SWA cache partitioning, C4/C128 behavior where used, page padding/alignment,
scale placement and index conversion. A shared dtype name or logical tensor shape is
not a physical-layout specification. Keep lengthy design explanations here; code retains
upstream comments/spacing with concise MACA difference notes.

## Tool binding

Include the spec in its source task's files. Before local source checks run, use
`model-review --id TASK --entry vllm_metax/models/.../model.py --spec .../SPEC.md`.
Repeat entries/specs for the models owned by the task. The tool runs the bundled static
audit and stores source/spec hashes and an audit artifact. Changed specs/imported local
helpers invalidate this review. Bind baseline `tools/models` samples to the corresponding
reviewed spec and test rows; neither a missing spec nor an unbound model is complete.
