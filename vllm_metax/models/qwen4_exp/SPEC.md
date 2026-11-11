# Qwen4Exp MetaX specification

## 1. Identity and scope

- Registered architectures: `Qwen4ExpForCausalLM`,
  `Qwen4ExpForConditionalGeneration`, `Qwen4ExpMTP`.
- Target vLLM `v0.33.0`,
  `4ba787ec9959e566d3b587a796dff07673324da5`; prior vLLM `v0.32.0`.
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
  Qwen3-VL remain external reusable model components from upstream because they
  are outside the Qwen4Exp model-owned directory.
- Top-level Qwen4Exp model construction is local. Import audit reports zero
  prohibited family imports and zero unresolved dynamic imports.

## 4. Weights and compute

- Sample `Qwen3.8-Flash-Next-dummy-3layers` has hidden size 2560, 24 attention
  heads, two KV heads, head dimension 256, QSA ratio four, four index heads,
  index head dimension 128, 512 experts, ten active experts, and one MTP layer.
- Preserve target PLE shard loading, FP8/NVFP4 table decoding, hyper-connection
  sharding, GDN state, QSA projections, and MTP mappings.
- C500 executes BF16 activations and supported W8A8 linears. NVFP4/FP8
  lookup branches remain source-adapted and unverified.

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

## 7. Chip scope and validation

- Bound sample: `Qwen3.8-Flash-Next-dummy-3layers`, TP4 dummy smoke.
- Source acceptance covers import audit, registration, QSA cache geometry and
  selector numerics, PLE gate numerics, and MTP dispatch.
- Final acceptance requires normal-entry prefill, two-token decode,
  concurrent batches, and `tools/batched_test --infer`.
- FP8/FP4 PLE and cache branches remain unverified on C500.

## 8. Open obligations

- Final model e2e and batched evidence are recorded in validation.
- Revisit NVFP4 and FP8 table/cache execution on supported hardware.
