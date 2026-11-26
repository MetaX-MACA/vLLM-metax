# MiniMax-M3 MetaX specification

## 1. Identity and scope

- Registered architectures: `MiniMaxM3SparseForCausalLM`,
  `MiniMaxM3SparseForConditionalGeneration`, `MiniMaxM3MTP`,
  `DSparkMiniMaxDraftModel`.
- Target vLLM `v0.34.0`,
  `62d4bc3c7e33b1b6c906e233e6af2d3506c482ca`; prior vLLM `v0.33.0` (
  `4ba787ec9959e566d3b587a796dff07673324da5`).
- `README.md` is a design note; this file is the canonical model spec.

## 2. Difference matrix

| Contract | Upstream CUDA | MACA C500 | Local implementation |
| --- | --- | --- | --- |
| Main sparse attention | MSA/CUTLASS | Local Triton sub-tile prefill/decode | `common/sparse_attention.py` |
| Indexer score/top-k | NVIDIA kernels | Local Triton score and block top-k | `common/indexer.py`, `common/ops/index_topk.py` |
| Index cache | BF16/FP8/NVFP4 capable | BF16 required on C500 | `metax/model.py` fused insert |
| Vision tower | NVIDIA selection | Mirrored local component with mRoPE setup | `common/vision_tower.py` |
| DSpark | NVIDIA DFlash lifecycle | Local MetaX lifecycle | `metax/dspark.py` |

The target ROCm context-parallel guard and NVIDIA MSA rejection are retained.
An SM100 capability alias never enables MSA on MACA.

## 3. Local model dependencies

- The complete target `vllm/models/minimax_m3/` tree is mirrored locally. The
  upstream AMD implementation is removed and the complete `nvidia/` tree is
  renamed to local `metax/`.
- `common/` retains the complete non-vendor indexer, sparse attention, vision
  tower, multimodal preprocessing and encoder graph implementation. Local
  `metax/` contains the complete NVIDIA-derived model and backend paths.
- No upstream concrete MiniMax or Qwen model is inherited or constructed.

## 4. Weights and compute

- Preserve checkpoint expert/linear maps, destination-dependent scales,
  hybrid load sessions, Gemma accumulation, and TP reductions.
- Gemma RMSNorm uses FlashInfer for supported unquantized inputs. The local
  Triton normalization handles BF16 rows and produces MXFP8 only for a
  compatible consumer. Its import does not initialize the CuTe indexer.
  The PyTorch fallback accumulates residuals in FP32.
- Preserve sigmoid routing, correction bias, shared experts, and dense/MoE
  layer selection.
- C500 uses BF16 and supported INT8 experts. CUDA-only NVFP4/MSA paths are not
  selected.
- `expert_dtype: "fp4"` checkpoints select group-32 E2M1/E8M0 routed experts on
  the local W4A16 path. Packed weights remain uint8, E8M0 scales preserve their
  exponent bytes, and every local gate/up/down shard is validated before kernel
  preparation. FP8 linears and shared experts remain on the FP8 path.

## 5. Attention and cache

- Main KV cache is `[blocks, num_kv_heads, block_size, 2 * 128]` with BF16
  storage on C500 and allocator-provided strides. K/V writers use token slot
  IDs; negative padding slots are skipped.
- The indexer key cache is key-only `[blocks, 1, block_size, 128]`, BF16 on
  C500. Index block size is 128 tokens and pages align with the main cache
  group.
- The fused Q/K normalization + RoPE + insert path writes main K/V and index K
  in one launch, then scores index rows and selects physical sparse blocks.
- The shared top-k buffer is
  `[round_up(max_num_batched_tokens, 4), num_index_heads / TP, topk_blocks]`
  `int32`. Token-major storage is transposed only for attention reads.
- Top-k block IDs are physical block IDs in stored-block units. Prefill,
  decode, mixed batches and speculative decode preserve split-K and causal
  sentinel handling.

- The fused cache-insert extension preserves in-place writes by default.
  `write_qkv=False` requires an FP8 Q destination and skips packed QKV stores.
  The wrapper checks the loaded operator schema before passing the flag.

## 6. Other execution contracts

- Preserve multimodal encoder graph state, mRoPE and video position paths.
- MTP and DSpark keep the same cache object, cache positions, and selected
  block IDs as the parent model.
- Dense MLP and shared-expert accumulation occur before the single required
  tensor-parallel reduction.

## 7. MACA porting recipe

Use the target MiniMax loader order and local sparse attention/indexer.
Preserve input aliases and the FP32 residual sum in Gemma RMSNorm. Use the
local Triton norm helper for eligible BF16 rows and select its MXFP8 output
only when the following GEMM accepts that format. Keep CuTe, MSA, NVFP4
indexer and grouped GEMMs behind their hardware checks and lazy imports.

Propagate `write_qkv` across the Python wrapper, stable Torch schema, C++
interface and kernel launch. With the default flag, write the rotated
packed QKV as before. When the caller passes False, require an FP8 Q
destination, skip packed QKV stores and still write the cache. Discover
optional support from the loaded operator schema so the wrapper can use the
older native signature for its default route.

Keep the target DSpark implementation in `common/dspark.py` and use
`metax/dspark.py` for the registered draft. For context KV, retain the
registered QKV quantization method. Apply the hidden norm, run each layer's
QKV projection and slice K/V after its Q width. Stack the layer results,
then apply each layer's Gemma K norm with `weight + 1`. The upstream
precomputation assumes ordinary RMSNorm weights. Keep that difference when
reapplying the cache fill and rollback positioning. Retain the local indexer format for the selected attention path.
The CUDA MSA/NVFP4 variants have separate index packing and scratch layouts
and must not reinterpret the C500 cache.
