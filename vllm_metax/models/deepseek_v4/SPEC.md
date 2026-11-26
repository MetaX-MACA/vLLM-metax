# DeepSeek V4 MetaX specification

## 1. Identity and scope

- Registered architectures: `DeepseekV4ForCausalLM`,
  `DeepSeekV4MTPModel`, and `DSparkDraftModel`.
- Target: vLLM `v0.34.0`,
  `62d4bc3c7e33b1b6c906e233e6af2d3506c482ca`.
- Prior base: vLLM `v0.33.0`,
  `4ba787ec9959e566d3b587a796dff07673324da5`.
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

Enable the packed FP8/FP4 implementations only with a chip and kernel
provider that support the exact cache record and arithmetic.

## 3. Local model dependencies

- Local `common/` is the complete non-vendor target mirror, including RoPE,
  partial-state storage, EPLB, multimodal preprocessing, sparse-MLA metadata,
  and common ops. MACA-only extensions live in `common/ops/*_metax` helpers.
- The complete NVIDIA model, MTP, DSpark, compressor, attention and helper
  implementations are under local `metax/`; they are adapted in place rather
  than imported from upstream.
- Public framework APIs under `vllm.model_executor.models.{utils,interfaces,
  registry,config,adapters}` and `vllm.models.common` remain upstream.
- Bind the registered entries in `deepseek_v4/__init__.py` to this local
  family, including the vision implementation and each configured drafter.

## 4. Weights and compute

- Sample `DeepSeek-V4-Flash-0731-smq-w8a8` uses BF16 runtime dtype,
  compressed-tensors dynamic INT8 token activations, channel INT8 weights,
  seven transformer layers, 64 Q heads, one KV head, head dimension 512,
  256 routed experts, six active experts, and one shared expert.
- Checkpoints may contain BF16, FP8, or FP4 expert weights. C500 execution is
  BF16/INT8; checkpoint storage does not imply FP8 or FP4 compute support.
- DSpark maps checkpoint `mtp.<i>` quantization prefixes to draft
  layers at `num_hidden_layers + i` and maps `mtp.0.main_proj`
  separately, before constructing quantized projections.
- Preserve fused WQ-A/WKV projections, grouped `wo_a`, sharded `wo_b`, EPLB
  logical-to-physical maps, shared experts, indexed layer arrays, and TP/PP/
  EP/DCP collective ownership. Grouped inverse RoPE and output projection run
  after attention and before MoE ownership changes.

## 5. Attention and indexer cache

### Upstream CUDA packed caches

For `fp8_ds_mla`, upstream `attention.py` publishes a 584-byte state.
`common/ops/cache_utils.py` stores all 576-byte data rows before the
scale region in each page. Each data row has 448 E4M3 NoPE bytes and 64
BF16 RoPE values (128 bytes). The scale region gives each row seven UE8M0
scales covering 64 values each and a padding byte. The page stride may add
padding to a multiple of 576. The reader computes the data and scale offsets
from the physical page and stored-state ordinal.

The upstream indexer stores 128 FP8 key values with one FP32 scale in a
132-byte row, or 64 packed MXFP4 bytes with four UE8M0 scales in a 68-byte
row. The indexer cache spec declares the compressed state ratio. Its CUDA
backend and the main cache dtype determine page alignment. On MACA the
132-byte row holds INT8 key values and an FP32 scale.

### MACA main/SWA BF16 cache

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

## 7. MACA porting recipe

Use the target attention wrapper to configure `DeepseekV4GroupedWoA`. Keep
the group count per TP shard and call `grouped_forward` before flattening into
`WoB`. The C500 path keeps the inverse RoPE output in BF16 and performs grouped
projection without the FP8 einsum. Keep the FP8 route gated. In particular,
LoRA requires the BF16 `WoA` input and therefore disallows fused RopeQuant.

Keep the compressor wkv/wgate linear merged and emit FP32 state via
`DeepseekV4CompressorKVScoreLinear`. Feed explicit token positions and allocator
block spans through the SWA and compressed cache paths. Short contexts must
have valid top-k rows even when all keys fit. Match inverse RoPE head packing
and output strides before replacing a backend.

Use the local qwen3_dspark helpers. Compute draft logits with `skip_gather=True`
for each TP rank and let the target `DraftLogitsCache` row mapping select the
logit rows. Allocate the pre-`hc_head` residual buffer only for drafters that
consume it. DSpark reads the auxiliary hidden states instead. A later decoder
layer can reuse the residual storage after attention has consumed it. The
first layer must treat the initial x alias as read-only.
