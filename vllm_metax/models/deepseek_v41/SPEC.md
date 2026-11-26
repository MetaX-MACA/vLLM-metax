# DeepSeek V4.1 MetaX specification

## 1. Identity and scope

- Registered architectures: `DeepseekV41ForCausalLM`,
  `DSparkDeepseekV4ForCausalLM`, and `DSparkV41DraftModel`.
- Target: vLLM `v0.34.0`,
  `62d4bc3c7e33b1b6c906e233e6af2d3506c482ca`.
- Prior base: vLLM `v0.33.0`,
  `4ba787ec9959e566d3b587a796dff07673324da5`.
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
- Replace upstream encoder attention selection with the local vision
  selector in `vllm_metax.models.vision`. Pass the selected backend into the
  model-local encoder graph setup so capture and replay agree on the backend.

## 4. Weights and compute

- Preserve target fused/shared-expert mappings, indexed layer arrays,
  checkpoint sharding, inverse RoPE, grouped `wo_a`, and TP/PCP collective
  ownership.
- `shard_fused_wqa_wkv` uses one token shard per TP rank followed by an
  all-gather from 2048 tokens. Cache-position propagation remains explicit
  across PCP ranks.
- Use BF16/INT8 on C500. Each FP8/MXFP8/NVFP4 checkpoint or cache branch
  requires a backend that supports its exact format.

## 5. Attention and indexer cache

### Upstream CUDA packed caches

In target `common/ops/cache_utils.py`, the model-specific widths select
the cache record. V4.1 MXFP8 stores 512 E4M3 bytes followed by sixteen
UE8M0 scales per row, one scale per 32 dimensions. The 528-byte rows
include the rotated RoPE values. V4.1 NVFP4 stores 256 packed E2M1 bytes with
the even value in the low nibble, followed by 32 E4M3 scales, one per 16
dimensions. This 288-byte format serves compressed KV. In both formats
each state's scales immediately follow its data, and the page uses the
allocator stride. The target's V4 compatibility record uses separate data
and scale regions as described in the [V4 recipe](../deepseek_v4/SPEC.md).

Target `attention.py` derives indexer widths independently. FP8 uses 128
E4M3 bytes plus one FP32 scale. MXFP4 uses 64 packed bytes plus four UE8M0
scales. On MACA the indexer uses symmetric INT8 in the 132-byte layout.

### MACA INT8 DS-MLA compressed and SWA cache

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

- PP intermediate tensors carry `input_ids` as int64 along with hidden states,
  residual and mixture state. Later stages restore the same token IDs before
  Engram hashing and token-aware expert routing. The first stage remains the
  token-ID producer; sequence-parallel token sharding requires PP=1.
- `supports_aux_hidden_states_over_pp=True` declares this intermediate-state
  contract.

## 7. MACA porting recipe

Start from the target V4.1 model, draft, replay and specialized-decode
entries as one unit. Select the INT8 KV dispatch in `attention.py` and let
`compressor.py` write to the same cache object that the local attention wrapper
reads. Pass logical cache positions and physical slot mappings separately
through PCP, the indexer and graph metadata. Keep ratio-1 and ratio-2 cache
source sharing and preserve SWA-only layers. Copy the target sparse metadata
contracts before substituting the CUDA packed record with the INT8 layout above.

Propagate `input_ids` across PP stage boundaries before Engram hash and token-aware
expert selection. Preserve the Engram fallback and host-table behavior behind
the target placement guards. Build replay metadata from the same per-layer
state as eager execution. Allocate replay graph scratch and specialized decode workspace only for
the selected graph and vendor paths.

The target's CUDA specialized-decode code includes CuTe top-k, NVFP4/MXFP8
kernels, distributed mHC boundaries and fused collectors. Keep imports lazy in
the local mirror. The local INT8 cache, per-request metadata and MACA sparse
attention remain the C500 execution path. Confirm the backend gate for each
vendor kernel before using it on a different chip.
