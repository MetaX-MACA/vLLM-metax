# Trimming by Model Structure

Field names are search hints. The runtime configuration classes, MetaX registrations,
and model implementations determine their meaning. First identify whether the effective
text configuration lives at the top level or under `text_config`, `language_config`, or another node.

## Homogeneous Dense Models and Standard Attention

- Depth may be expressed as `num_hidden_layers`, `n_layer`, `num_layers`, or a property alias.
  Edit the actual source field and verify that normalization preserves the intended value.
- Prefer depth-only reductions. Keep the tokenizer, `vocab_size`, special token IDs, and
  embedding/lm_head compatible. Do not shrink the vocabulary just to save memory; added
  token IDs must also remain within the embedding range.
- Before narrowing dimensions, check Q/K/V projections, head_dim, GQA/MQA Q-to-KV head
  relationships, TP sharding, and fused-layer sizes. Do not treat
  `hidden_size = heads * head_dim` as universal; MLA and other architectures may define
  independent projection dimensions.
- Preserve RoPE and positional encoding settings where possible, limiting runtime budgets
  with `--max-model-len`. Consider structural edits if the model itself allocates large
  buffers from its configuration.

## MoE

- `first_k_dense_replace`, `moe_layer_freq`, `mlp_layer_types`, sparse strides, or layer-index
  modulo rules may jointly select Dense/MoE layers. Keeping only early layers can bypass experts entirely.
- Retain at least one required Dense and MoE layer and the target shared-expert path.
  Recalculate boundaries and periods using new layer indices rather than editing descriptive arrays alone.
- When reducing experts, inspect the consumers of `n_routed_experts`/`num_experts`, enforce
  `num_experts_per_tok <= experts`, and check `n_group`/`topk_group` and the number of eligible
  experts under grouped routing. Check EP partitioning, redundant experts/EPLB, shared
  experts, and implementation-specific TP constraints. Random-weight requests cannot
  guarantee execution of every expert or routing branch.

## MLA, Sparse Attention, and Compressed Caches

- Preserve q/kv/o LoRA ranks, nope/rope/v head dimensions, indexer dimensions, and
  quantization unless new shapes are verified against current kernels. These dimensions
  often determine whether a particular implementation is usable.
- Distinguish layers that compute indices from layers that reuse them. Preserve producer
  ordering and dependencies. Check patterns, frequencies, offsets, global layer indices,
  and PP boundaries; shared layers must not reference removed state.
- Preserve target combinations of `compress_ratios` and sliding-window/full/sparse layers.
  Select per-layer array values using the mapping. If an array includes prediction layers
  or extra slots, inspect consumer indexing first. Not every list must equal backbone depth;
  do not trim unrelated lists such as vocabulary IDs or RoPE parameters.
- Requests must cross the relevant `index_topk`, sliding-window, or compression boundary.
  Short-input success does not establish sparse-selection coverage.

## Hybrid Attention / SSM / Mamba / Linear Attention

- Inspect `layer_types`, `layers_block_type`, periodic fields, and logic deriving types from
  layer indices. Retain required attention and state-space layers and their state/cache paths.
- Check SSM state/head/group dimensions, convolution kernels, and TP constraints. Distinguish
  prefill state initialization from decode updates. Do not claim hybrid coverage if changing
  a period leaves only one layer type.
- Apply the same reasoning to alternating sliding-window and full attention. Cross-layer
  KV sharing also requires synchronized producer mappings and consumer references.

## Multimodal and Encoder-Decoder Models

- Inspect text, vision, audio, encoder, decoder, projector, and resampler configurations
  separately. Reduce text depth first. Reducing encoder depth requires updating feature
  selection, including negative indices, concatenated feature counts, projection input
  sizes, and interfaces between modalities.
- Do not recursively reduce every hidden size or layer count. Preserve compatibility of
  independent encoder/decoder depths, cross attention, decoder start tokens, shared
  embeddings, and vocabularies.
- Retain processor/preprocessor, image/audio preprocessing, and modality-token configuration.
  Copy local code referenced by `auto_map` and its relative imports; custom tokenizers have
  the same requirement. Use required, trusted remote code. Offline mode cannot supply missing assets or code.
- Text-only requests do not prove vision/audio encoder execution. Provide at least one valid
  input for each target modality. Respect processor minima when reducing image resolution
  or audio duration, and record reduced coverage.

## Quantization, MTP, and Other Layer References

- Quantization may live in `quantization_config`, `compression_config`, or separate files.
  Preserve the method, weight/activation dtypes, group/block sizes, packing, and dynamic/static scale semantics.
- Update exact module-name layer indices in ignore/targets/skip/modules_to_not_convert and
  per-layer overrides. Preserve exclusions for gates, indexers, lm_head, and other modules.
  Replace layer-index path segments precisely, not digits globally. Handle regex and
  wildcard entries according to their matching semantics.
- MTP/nextn layers may follow backbone indices or use an independent `mtp.*` namespace.
  Update prediction-layer configuration, quantization rules, and draft/target references
  together. Do not automatically zero their counts or preserve obsolete backbone offsets.
  Retaining MTP configuration does not make ordinary serving execute it; explicitly enable
  and validate speculative decoding.
- For DSpark and other intermediate-layer consumers, remap target IDs and check their range,
  order, count, and prediction-head contract. If dependencies are removed, state that the
  feature is no longer covered; do not silently substitute targets to hide invalid references.

## Existing Repository Examples

These are configuration observations made when creating this skill, not support guarantees
across versions. Reread the actual files when using an example.

### GLM-5_2-W8A8-dummy-3layers

Location: `tools/batched_test/models/GLM-5_2-W8A8-dummy-3layers/`.
Its README records structural layer mappings `0→0, 6→1, 7→2`, with MTP layer `78→3`.

- Three backbone layers, `first_k_dense_replace=1`, and `mlp_layer_types=[dense,sparse,sparse]`.
- `indexer_types=[full,full,shared]` and explicit `index_topk_pattern="FFS"`.
  The current `vllm_metax/models/deepseek_v2.py` reads the pattern first, falling back to
  frequency/offset only when absent. Editing `indexer_types` alone does not control this path.
- Preserves hidden_size=6144, 256 routed experts, top-k=8, MLA dimensions, and W8A8.
  Quantization ignore entries follow the layer mapping; `num_nextn_predict_layers=1` remains.
- With `index_topk=2048`, short requests cover basic execution only. Sparse selection needs
  a separate request exceeding that threshold.
- Historical README results specify an environment and untested scope; they are not evidence
  that the current serve/TP/MTP tests have passed.

### DeepSeek-V4-Flash-0731-smq-w8a8

Location: `tools/batched_test/models/DeepSeek-V4-Flash-0731-smq-w8a8/`.

- Seven backbone layers, hidden_size=2048, 256 routed experts, top-k=6, and `num_hash_layers=3`.
  `metax/model.py` selects the MoE path by comparing layer indices with the hash-layer boundary;
  consider both sides of that boundary when trimming.
- `compress_ratios` has ten entries; the first seven are `[0,0,4,128,4,128,4]`.
  Current `models/deepseek_v4/attention.py` indexes backbone entries and applies `max(1, ratio)`,
  using 1 for MTP. Other cache/backend consumers may traverse the entire list; inspect all
  consumers before edits. Do not automatically reject ten entries or truncate them to seven.
- DSpark references `[4,5,6]`, with `num_nextn_predict_layers=1`. Check both when changing depth.
- `expert_dtype="fp4"` coexists with compressed-tensors W8A8 metadata. One field alone cannot
  establish effective expert/projection dtypes; inspect MetaX dispatch.
- `run_serve2.sh` shows dummy + DSpark startup intent but uses an old `./temp/...` path and
  specific environment variables. Generate commands with current paths; script existence
  is not a successful test record.
