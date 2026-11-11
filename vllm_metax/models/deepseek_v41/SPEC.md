# DeepSeek V4.1 MetaX specification

## 1. Identity and scope

- Registered architectures: `DeepseekV41ForCausalLM`,
  `DSparkDeepseekV4ForCausalLM`, and `DSparkV41DraftModel`.
- Target: vLLM `v0.33.0`,
  `4ba787ec9959e566d3b587a796dff07673324da5`.
- Prior base: vLLM `v0.32.0`,
  `5df63a7203cdb9328d824a27b682e77c7b9928be`.
- The complete target `vllm/models/deepseek_v41/` tree is mirrored locally.
  The upstream AMD tree is removed and the complete `nvidia/` tree is renamed
  to local `metax/`.

## 2. Difference matrix

| Contract | Upstream CUDA | MACA C500 | Local implementation and reason |
| --- | --- | --- | --- |
| Compressed KV | FP8/MXFP8 or NVFP4 records | 576-byte INT8 states | `metax/ops/kv_cache_int8.py` |
| SWA KV | Packed target record | Same 576-byte INT8 state format | `attention.py` dispatch |
| Sparse MLA | FlashMLA sparse/mega-attention | Local wrappers and sparse metadata | `metax/flashmla.py` |
| Indexer K | FP8/MXFP4 store | INT8 values with FP32 block scale | local `metax/ops/kv_cache_int8.py` with upstream store helper |
| PCP and replay | Target collectives/graph | Explicit cache position/slot gather and MetaX replay wrappers | `attention.py`, `decoder_replay_layers.py` |

NVIDIA-only mega-attention modules are copied for source completeness and
capability-gated. They are not selected on MACA.

## 3. Local model dependencies

- `common/` is the complete non-vendor target mirror, including decoder replay,
  RoPE, layout, query-quant, indexer-store, Engram, multimodal and vision
  helpers. Model-family imports resolve to this local mirror.
- `metax/` contains the complete NVIDIA-derived model, draft, replay,
  specialized-decode, FlashMLA and helper implementation with MACA changes.
- `models.deepseek_v41.quant_config` remains local for the MACA quant config.
- Shared public APIs remain under `vllm.model_executor.models` and
  `vllm.models.common`.
- The import audit reports zero prohibited concrete-model imports and zero
  unresolved dynamic imports.

## 4. Weights and compute

- Preserve target fused/shared-expert mappings, indexed layer arrays,
  checkpoint sharding, inverse RoPE, grouped `wo_a`, and TP/PCP collective
  ownership.
- `shard_fused_wqa_wkv` uses one token shard per TP rank followed by an
  all-gather from 2048 tokens. Cache-position propagation remains explicit
  across PCP ranks.
- C500 execution is BF16/INT8. FP8/MXFP8/NVFP4 checkpoint and cache branches
  remain source-adapted but unverified.

## 5. Attention and indexer cache

### INT8 DS-MLA compressed and SWA cache

- A paged state has 576 bytes: 512 INT8 value bytes and 64 bytes of scales.
- Values occupy `[0, block_size * 512)`; scales occupy the following
  `block_size * 64` bytes. This is block-segregated, not interleaved by token.
- Values are 448 NoPE plus 64 RoPE dimensions. There are 16 FP32 scales per
  state: 14 for NoPE and two for RoPE; each scale covers 32 dimensions.
- Quantization is symmetric with `scale = max(abs(x)) / 127`, floor
  `2^-30`, round-half-away-from-zero, clamp `[-127,127]`.
- `stride(-1) == 1`, `stride(-2) == 576`; `stride(0)` may include allocator
  padding. Page alignment is 576 bytes. Block size is expressed in stored
  states.

### Indexer cache

- A row is 132 bytes: 128 INT8 index-key values plus one FP32 scale.
- The indexer page shares the physical block with main KV pages and carries
  the packed KV page alignment. Index and block-table units are stored states.

### Execution

- SWA-only layers use ratio 0 and do not own a compressed cache.
- Ratio-1 and ratio-2 layers share the kv-source compressed cache and use
  block sizes `attention_block_size / 1` or `/ 2`.
- Prefill gathers use explicit `cache_slot_mapping` and `cache_positions`;
  decode and mixed batches share the same top-k, indexer and cache slot units.
- MTP/DSpark consume the same cache objects and explicit positions.

## 6. Other execution contracts

- Preserve target layer topology, dense/MoE transitions, shared experts,
  sequence-parallel padding, and pipeline boundaries.
- The decoder replay path must use the same local layer classes and cache
  positions as eager execution; replay must not rebind an upstream decoder.

## 7. Chip scope and validation

- Current scope: MetaX C500 source mode, BF16/INT8.
- No V4.1 sample exists under `tools/models/`; model-level e2e is not claimed.
- Source acceptance requires import audit, registration, INT8 cache
  insert/gather numerics, compressor/cache geometry, and replay metadata
  contracts.
- FP8/FP4 branches remain unverified on C500.

## 8. Open obligations

- Add and bind a V4.1 dummy or real checkpoint sample before claiming
  model-level V4.1 validation.
- Validate INT8 compressed, SWA-only, ratio-1 and ratio-2 layers in final
  model runs when a sample is available.
