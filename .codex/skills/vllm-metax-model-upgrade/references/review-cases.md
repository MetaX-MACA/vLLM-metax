# Model Upgrade Review Cases

Use only the cases relevant to the requested model. These are historical failure
patterns, not assertions about the currently installed components. Recheck target
upstream and MACA behavior before making changes or support declarations.

## Quantization and loading

- Distinguish checkpoint storage, runtime weights, activations, KV cache and indexer
  cache dtypes. A model described as FP8 need not use FP8 for all five.
- Follow fused weight mappings, shard loaders, E8M0 decoding, scale orientation and
  post-load repacking through to the tensor actually consumed by the kernel.
- For WK dequantization, derive row and column groups independently when the format
  permits shape-based inference. Validate per-channel scale length and actual block
  metadata. Equal dtype does not imply square groups; padded tail groups cannot be
  inferred by an unconditional exact-division rule. Test weight/scale arrival order.
- Check missing required weights separately from optional scalar cache scales. Preserve
  intentional loader validation contexts. Loaded optional parameters must not make a
  completeness check index an empty missing-parameter set.

## DeepSeek V4 and model-private attention

- Inventory attention, compressor, indexer, SWA/compressed cache, inverse RoPE, grouped
  output projection, router and MTP/DSpark together. Follow upstream moves recursively.
- Establish actual MACA BF16/INT8/FP8/FP4 choices before porting a CUDA path. Verify
  cache byte layout, scales, page padding, C4 overlap and C128 compression as relevant.
- Validate decode and prefill separately, including mixed batches, padded heads,
  request-relative vs global indices, causal boundaries and attention sinks.
- Treat `SupportsLoRA` as an interface claim, not proof. Trace whether each adapted
  target executes its wrapper or reads base weights directly. Grouped `wo_a` einsum
  can bypass LoRA deltas in upstream as well as locally; compare both before attribution.
  The architecture's `o_lora_rank` alone does not mean adapter LoRA is being applied.
- Inspect installed MegaMoE APIs and hardware restrictions instead of copying NVIDIA
  support claims. Optional shared-expert arguments need actual signature/semantic
  support; a fallback must not pass new keywords to an old API. Distinguish unavailable
  optional MegaMoE from the ordinary FusedMoE path.

## Parallelism, model identity and draft layers

- Follow all-reduce/reduce-scatter/all-gather ownership across attention, normalization,
  residual and MoE. A later reduce-scatter must not sum an already all-reduced result.
  Include TP size, PP boundaries, SP padding and the final dense/MoE choice.
- Do not identify a model-specific MTP exception only by layer count. Trace config
  identity through platform and speculative rewrites, preserving an explicit identity
  when needed; compute SP flags after determining the actual MLP implementation.
- Confirm main and draft models use the intended local/upstream decoder, buffers,
  parameters and weight mappings. Imports with identical class names may bind to
  different implementations. Verify registration and startup, not just file presence.
