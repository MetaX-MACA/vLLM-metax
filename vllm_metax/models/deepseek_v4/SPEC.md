# DeepSeek V4 MetaX specification

## 1. Identity and scope

- Registered architectures: `DeepseekV4ForCausalLM`,
  `DeepSeekV4MTPModel`, and `DSparkDraftModel`.
- Target: vLLM `v0.33.0`,
  `4ba787ec9959e566d3b587a796dff07673324da5`.
- Prior base: vLLM `v0.32.0`,
  `5df63a7203cdb9328d824a27b682e77c7b9928be`.
- Local entries: `__init__.py`, `metax/model.py`, `metax/mtp.py`,
  `metax/dspark.py`, `metax/flashmla.py`, `metax/gate_linear.py`.
- The complete target `vllm/models/deepseek_v4/` model tree is mirrored locally.
  The upstream AMD/XPU/CPU implementations are removed and the complete
  `nvidia/` tree is renamed to local `metax/`.

## 2. Difference matrix

| Contract | Upstream CUDA | MACA C500 | Local implementation and reason |
| --- | --- | --- | --- |
| Main KV cache | Packed FP8 DS-MLA `uint8` records | Plain BF16 rows | `MacaDeepseekV4Attention`; C500 does not execute FP8 cache arithmetic |
| Sparse indexer cache | FP8 or MXFP4 packed rows | INT8 `uint8` rows with FP32 scales | `MacaDeepseekV4Indexer`, `fused_indexer_q_rope_int8_quant` |
| Compressor | CUDA CuTeDSL/Triton | BF16 or INT8 local Triton | `MacaDeepseekCompressor` |
| Sparse MLA | CUDA FlashMLA | MetaX FlashMLA wrapper and Triton gathering | `metax/flashmla.py` |
| Router GEMM | Target CUTLASS/CuteDSL tiers | FP32 `_C` router kernel plus `torch.mm` epilogue | `metax/gate_linear.py` |
| MegaMoE | SM100 DeepGEMM | Ordinary MoE only; unsupported MegaMoE rejects | `metax/model.py` |

The packed FP8/FP4 implementations remain source-adapted but are unverified on
C500. Removal is conditional on a supported chip and installed component
providing the exact target record and arithmetic.

## 3. Local model dependencies

- Local `common/` is the complete non-vendor target mirror, including RoPE,
  partial-state storage, EPLB, multimodal preprocessing, sparse-MLA metadata,
  and common ops. MACA-only extensions live in `common/ops/*_metax` helpers.
- The complete NVIDIA model, MTP, DSpark, compressor, attention and helper
  implementations are under local `metax/`; they are adapted in place rather
  than imported from upstream.
- Public framework APIs under `vllm.model_executor.models.{utils,interfaces,
  registry,config,adapters}` and `vllm.models.common` remain upstream.
- `audit_imports.py` reaches all local helpers from `deepseek_v4/__init__.py`
  and reports zero prohibited concrete-model imports and zero unresolved
  dynamic imports. Same-family `vllm.models.deepseek_v4...` imports and the
  removed AMD/XPU/CPU trees are prohibited.

## 4. Weights and compute

- Sample `DeepSeek-V4-Flash-0731-smq-w8a8` uses BF16 runtime dtype,
  compressed-tensors dynamic INT8 token activations, channel INT8 weights,
  seven transformer layers, 64 Q heads, one KV head, head dimension 512,
  256 routed experts, six active experts, and one shared expert.
- Checkpoints may contain BF16, FP8, or FP4 expert weights. C500 execution is
  BF16/INT8; checkpoint storage does not imply FP8 or FP4 compute support.
- Preserve fused WQ-A/WKV projections, grouped `wo_a`, sharded `wo_b`, EPLB
  logical-to-physical maps, shared experts, indexed layer arrays, and TP/PP/
  EP/DCP collective ownership. Grouped inverse RoPE and output projection run
  after attention and before MoE ownership changes.

## 5. Attention and indexer cache

### Main/SWA BF16 cache

- Shape is `[num_blocks, block_size, 512]`, BF16, one KV head.
- The last dimension is 448 NoPE plus 64 RoPE values; it is contiguous and
  has no packed scale bytes. A token row is 1024 bytes.
- The cache spec declares BF16 and alignment 512. The allocator may pad the
  physical page/block stride; producers and consumers use `stride(0)`.
- Compressed caches store `tokens_per_state == 1`, `4`, or `128` as selected
  by `compress_ratios`; compressed block size is the attention block size
  divided by the ratio. SWA-only layers reuse the shared SWA cache.

### Lightning-indexer INT8 cache

- Pooled row width is 132 bytes: 128 INT8 values followed by one FP32 scale.
- The scale covers all 128 values. The packed cache is `uint8`, shape
  `[num_blocks, block_size / compress_ratio, 132]`; block size is in stored
  states, while slot and gather indices are also stored-state units.
- `fused_indexer_q_rope_int8_quant` applies interleaved GPT-J RoPE to the last
  64 dimensions, rounds RoPE through BF16, performs symmetric INT8
  round-half-away-from-zero quantization, and folds the Q scale into the
  index weights.
- The pool-owned indexer cache intentionally has no page alignment. The
  indexer page is a subpage of the shared physical block and retains the
  allocator-provided block/head extent.

Allocation, compressor write, gather, sparse top-k, prefill/decode split,
mixed batches, C4/C128 compression, SWA, and MTP/DSpark use these same
physical block and logical slot units. Top-k/gather workspace width is
`round_up(top_k + window_size + max_image_tokens, 128)` and is `int32`.

## 6. Other execution contracts

- Router shapes include `(256,5120)` through 48 tokens, `(352,4096)`,
  `(256,8192)`, and the retained MetaX `(256,6144)` path.
- DSpark block size in the baseline sample is five. MTP owns its draft block,
  inverse-RoPE/grouped projection state, and cache-position lifecycle across
  load and replay.
- Sequence-parallel gathers and reductions preserve the target order: MoE may
  consume local tokens and return local outputs; callers perform any required
  final gather or reduction exactly once.

## 7. Chip scope and validation

- Current scope: MetaX C500, source kernel provider,
  `USE_PRECOMPILED_KERNEL=0`, BF16/INT8.
- Required sample: `DeepSeek-V4-Flash-0731-smq-w8a8`, TP1, dummy smoke.
- `DeepSeek-V4-Flash-FP8_sgl_project` is an explicit unsupported C500
  rejection row. FP8/FP4 compute and cache branches remain unverified.
- Source acceptance covers import closure, registration, indexer/cache
  geometry and numerics. Final acceptance additionally requires normal-entry
  prefill, at least two decode tokens, concurrent requests, and
  `tools/batched_test --infer`.

## 8. Open obligations

- Final model e2e and batched evidence are recorded only in validation.
- Revisit FP8/FP4 cache and compute on hardware and components with explicit
  support for the corresponding record and arithmetic.
