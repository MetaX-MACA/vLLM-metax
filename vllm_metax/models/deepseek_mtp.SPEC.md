# DeepSeek MTP

## 1. Identity and scope

- Registered architectures: `DeepSeekMTPModel`, `DeepseekV32MTPModel`.
- Target vLLM `v0.34.0`,
  `62d4bc3c7e33b1b6c906e233e6af2d3506c482ca`; prior vLLM `v0.33.0` (
  `4ba787ec9959e566d3b587a796dff07673324da5`).
- Implementation is local `deepseek_mtp.py` and binds draft blocks to the
  local DeepSeek decoder/indexer implementation.

## 2. Difference matrix

| Contract | Upstream CUDA | MACA C500 | Local implementation |
| --- | --- | --- | --- |
| FP8 indexer WK load | Two-step weight/scale arrival | Persistent pending pair across `load_weights()` | model-owned buffers |
| Draft attention/cache | Target CUDA | Same local BF16/INT8 cache contract as parent | local DeepSeek V2/V3 helpers |
| MTP top-k lifetime | Target skip/iteration rules | Explicitly initialized for every MTP layer | local decoder/indexer |

## 3. Local model dependencies

- Uses local `deepseek_v2.py` and shared public framework model interfaces.

## 4. Weights and compute

- FP8 WK weight and scale may arrive in either order. The pending buffer is
  model-owned and must survive repeated `load_weights()` invocations.
- Preserve draft PP mappings, shared embeddings, norm/projection weights,
  fused projections, and post-load transforms.
- C500 executes BF16/INT8; FP8 storage handling does not claim FP8 arithmetic.

## 5. Attention and indexer cache

- MTP consumes the parent model's MLA cache layout and C500 BF16/INT8 sparse
  indexer contract.
- Cache positions and top-k buffer writes are explicit for speculative
  iterations. A rejected draft row must not overwrite a committed indexer key
  needed by the next accepted step.
- MTP layers never begin with a skipped/uninitialized indexer top-k buffer.

## 6. Other execution contracts

- Preserve MTP pipeline boundaries, shared head/norm ownership, and draft
  layer indexing.
- The loader distinguishes missing required WK weights from
  optional runtime cache scales.

## 7. MACA porting recipe

Start with the target `deepseek_mtp.py` draft loader. Use local DeepSeek decoder
layers and use the target `get_fused_shared_expert_layers` helper with the actual
EPLB flag. Apply shared-expert fusion only to the designated draft layers.
Keep the checkpoint-to-draft mapping indexed from `num_hidden_layers` and the
local indexer WK fused projection handling. Return loaded parameter names to
the target loader for completeness checking. Preserve the parent DSA cache
geometry and hidden-state input when updating the draft entry.
