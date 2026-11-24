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
   evidence. Record unchanged non-vendor files as part of the local mirror. List the
   shared upstream modules and symbols reused from outside the model directory and why
   their contracts work on MACA. Record the layout explicitly when upstream has no
   `nvidia/` directory.
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

## Upstream component reuse

When a plugin-owned model reuses a component from another upstream model, add a section
with this exact heading to the consuming model spec. Start the section with a fenced JSON
list. Each declaration names a reviewed local entry, the local file holding a named
`from ... import ...` statement, the fully qualified upstream symbol, the full reviewed
vLLM commit and the use. Paths are relative to the MetaX source root. Add a declaration for
each entry and importing file that reaches the symbol.

For example, this template describes the QSA and ordinary full-attention uses in one local
Qwen4 Exp entry. Replace the commit placeholder after upstream review.

```json
[
  {
    "entry": "vllm_metax/models/qwen4_exp/__init__.py",
    "file": "vllm_metax/models/qwen4_exp/metax/qsa.py",
    "symbol": "vllm.model_executor.models.qwen3_next.Qwen3NextAttention",
    "upstream_revision": "<full reviewed vLLM commit>",
    "use": "Inherit _project_qkv_gate for QSA without fused preparation"
  },
  {
    "entry": "vllm_metax/models/qwen4_exp/__init__.py",
    "file": "vllm_metax/models/qwen4_exp/metax/model.py",
    "symbol": "vllm.model_executor.models.qwen3_next.Qwen3NextAttention",
    "upstream_revision": "<full reviewed vLLM commit>",
    "use": "Construct ordinary full attention when QSA is disabled"
  }
]
```

Below the list, explain why the symbol can be reused, which private dependencies and
side effects were reviewed, and which checkpoint/configuration rows exercise each use.
Keep the entry point, whole-model construction and same-family implementation local.
Declarations apply to named symbol imports. Module imports, wildcard imports and dynamic
lookups follow the standard audit rules.

Declare `e2e` and `batched_test` for the review task. For each use, bind the consuming
configuration to support-matrix rows that assert the component's worker origin, execution
branch and actual MACA backend. For Qwen4 Exp, verify QSA with fused preparation disabled
and ordinary full attention with QSA disabled. Record current evidence or mark the affected
configuration unverified until its obligations can be met.

## Tool binding

Include the spec in its source task's files. Before local source checks run, use
`model-review --id TASK --entry vllm_metax/models/.../model.py --spec .../SPEC.md`.
Repeat entries for all registered variants and specs for each model owned by the task.
The tool checks reuse declarations against the selected vLLM checkout and records matched
symbol imports in the audit artifact. When calling `audit_imports.py` directly, supply
`--spec LOCAL_SPEC.md` for each reviewed spec and `--upstream-source VLLM_CHECKOUT` for
the selected commit. Changed specs or imported local helpers invalidate the review.
Bind baseline `tools/models` samples to the corresponding
reviewed spec and test rows; neither a missing spec nor an unbound model is complete.
