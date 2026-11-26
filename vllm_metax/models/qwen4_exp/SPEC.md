# Qwen4Exp MetaX specification

## 1. Identity and scope

- Registered architectures: `Qwen4ExpForCausalLM`,
  `Qwen4ExpForConditionalGeneration`, `Qwen4ExpMTP`.
- Target vLLM `v0.34.0`,
  `62d4bc3c7e33b1b6c906e233e6af2d3506c482ca`; prior vLLM `v0.33.0` (
  `4ba787ec9959e566d3b587a796dff07673324da5`).
- Scope includes QSA attention/index selection, PLE and n-gram embedding,
  hyper-connections, GDN, text/vision state, and MTP.

## 2. Difference matrix

| Contract | Upstream CUDA | MACA C500 | Local implementation |
| --- | --- | --- | --- |
| QSA selector | NVIDIA QSA kernels | Local Triton paged selector | `metax/ops/qsa_indexer*.py` |
| QSA side caches | Target paged layout | BF16 ring and compressed caches | `common/qsa_cache.py` |
| Hyper-connection SP | Target automatic rule | Same v0.33 rule with local projection | `metax/hyperconnection.py` |
| Low-latency GEMM | NVIDIA CuTe-DSL | Disabled on MACA | `metax/low_latency_gemm.py` |

The target dtype-equality guard between Q and compressed K and 64-column
logits rounding remain active.

## 3. Local model dependencies

- The complete target `vllm/models/qwen4_exp/` model tree is mirrored under
  `vllm_metax/models/qwen4_exp/`. The upstream `amd/` implementation is removed;
  the complete `nvidia/` tree is renamed to local `metax/` and all model-family
  imports are rewritten to this local mirror.
- `common/`, `common/ops/`, model entry/config code and the renamed `metax/`
  implementation are retained even where unchanged. Qwen3.5, Qwen3Next and
  Qwen3-VL resolve through local model modules outside the Qwen4Exp directory.
  Model construction, ordinary full attention and MTP use local entries.

## 4. Weights and compute

- Sample `Qwen3.8-Flash-Next-dummy-3layers` has hidden size 2560, 24 attention
  heads, two KV heads, head dimension 256, QSA ratio four, four index heads,
  index head dimension 128, 512 experts, ten active experts, and one MTP layer.
- Preserve target PLE shard loading, FP8/NVFP4 table decoding, hyper-connection
  sharding, GDN state, QSA projections, and MTP mappings.
- C500 uses BF16 activations and supported W8A8 linears. NVFP4/FP8
  lookup requires the corresponding table backend.

## 5. Attention and cache

- Main paged K/V cache is BF16, shape `[blocks, kv_heads, states, 2 *
  head_dim]`; QSA reads the transposed `[blocks, states, kv_heads, ...]` view
  and splits K/V over the last dimension.
- Main cache block size is runtime-configured and must be a multiple of 16.
  QSA rejects FP8 main cache and fused output quantization on this path.
- Raw QSA key rows are BF16 vectors of `indexer_head_dim=128`. When MRoPE is
  used, exact three-axis int64 positions follow the key at
  `round_up(128, 4)` BF16 offsets, consuming 12 BF16 positions.
- The raw-key circular buffer capacity is `round_up(compress_ratio +
  num_speculative_tokens, compress_ratio)`. For the sample ratio four, capacity
  is eight for four speculative tokens and remains divisible by the attention
  block size.
- Compressed QSA keys store one BF16 row per completed group. Their page
  pitch is `block_size / ratio * head_dim * 2` plus a 3072-byte layer tail;
  the aggregate BLNHC pitch remains aligned for shared physical subpages.
- Producer/consumer units are: logical token position -> group boundary ->
  compressed row/block index -> physical page from block table -> BF16 row.
  Prefill, decode, mixed batches and MTP use the same explicit positions.

## 6. Other execution contracts

- PLE gate operates on BF16/FP16 `key/value/hidden`; `hidden` contains
  `hyper_connection` repeated groups. It normalizes key/query, computes the
  signed square-root gate, applies the value, then normalizes the convolution
  output.
- Hyper-connection sequence-parallel gathers are applied exactly once before
  MoE/local ownership changes.
- N-gram embedding preserves target hash arithmetic and int64 overflow
  semantics.

## 7. MACA porting recipe

Mirror the target QSA attention, common indexer and cache types. Make the
local `QSAIndexer` share target metadata with the Triton top-k fallback. Carry
Q and compressed K in the same dtype, preserve the int32 index units and
round the logits width to the target 64-column multiple. Use local
`qwen3_next.Qwen3NextAttention` for ordinary full attention and the projection
method QSA uses before cache preparation. Keep local Qwen3.5 and Qwen3-VL
model components for GDN and multimodal dispatch.

Build PLE/n-gram tables using the target sharding, hash and padding contracts.
Keep local lookup/normalization and use a stream only where the projection
ordering permits it. Apply the target SP rule once before the hyper-connection
reduction. Map the draft layer and output head to the corresponding local
Qwen4Exp MTP, preserving state tensors when switching target worker APIs.

## Native NIXL transfer contract

QSA advertises KV connector support only for the built-in `NixlConnector`.
Configuration validation runs before checkpoint loading and requires GPU cache
buffers, PP=1, hybrid cache management and DS convolution state layout. Sequential
Qwen4Exp MTP is accepted only with the matching draft architecture; parallel drafts
and custom connectors require a separate implementation. Both endpoints must have
matching cache-owner and MTP configurations. The native handshake checks cache compatibility.

QSA, GDN and replicated PLE state remain cache owners. The native NIXL contract preserves
recurrent/circular caches and excludes speculative scratch slots.
