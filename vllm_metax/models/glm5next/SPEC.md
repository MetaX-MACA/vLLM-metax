# GLM-5 Next MetaX specification

## 1. Identity and scope

- Registered architectures: `Glm5NextForCausalLM`,
  `Glm5NextForConditionalGeneration`.
- Target vLLM `v0.33.0`,
  `4ba787ec9959e566d3b587a796dff07673324da5`; prior vLLM `v0.32.0`.
- Scope includes the text backbone, vision tower/processor, KDA hybrid
  attention, K-pool sparse indexer, MTP, and mHC.

## 2. Difference matrix

| Contract | Upstream CUDA | MACA C500 | Local implementation |
| --- | --- | --- | --- |
| K-pool index cache | FP8/MXFP4 | INT8 128-byte row plus FP32 scale; BF16 tail | `metax/kpool_int8.py`, `metax/kpool_bf16.py` |
| Sparse attention | CUDA sparse MLA | Local metadata and BF16/INT8 cache readers | `metax/sparse_indexer.py` |
| KDA | NVIDIA kernels | Local KDA integration | `common/kda.py`, `metax/ops/third_party/kda` |
| mHC | CUDA fused path | Local TileLang/framework path | shared mHC kernels |

The target top-k width contract remains `round_up(index_topk + index_kpool -
1, 128)`. The incomplete trailing pool is mandatory.

## 3. Local model dependencies

- The complete target `vllm/models/glm5next/` tree is mirrored locally. The
  upstream AMD implementation is removed and the complete `nvidia/` tree is
  renamed to local `metax/`.
- `common/` retains the full non-vendor model, attention, KDA, multimodal, MTP
  and sparse-indexer implementation. The NVIDIA-derived K-pool/KDA and model
  implementation is under local `metax/`.
- No upstream concrete GLM or DeepSeek model is inherited or constructed.
  Import audit reports zero prohibited and zero unresolved dynamic imports.

## 4. Weights and compute

- Sample `GLM-5.3-Flash-dummy-3layers-w8a8` has three text layers, hidden size
  4096, 64 Q/KV heads, QK/V dimensions 256, 288 routed experts, eight active
  experts, one shared expert, `index_topk=2048`, and `index_kpool=4`.
- Runtime activation/weight quantization is compressed-tensors W8A8.
- Preserve fused routing, shared experts, KDA convolution/state mappings,
  multimodal projection, checkpoint names, and post-load repacking.
- C500 executes BF16/INT8. FP8/FP4 branches are source-reviewed but unverified.

## 5. Attention and indexer cache

- Sparse K-pool pages store all 128-byte INT8 quantized rows first and one
  FP32 scale per row after the value region. Rows are their pool slots in
  stored-state units.
- A pool is `index_kpool` consecutive token keys. The INT8 path applies
  Hadamard-128 rotation, materializes BF16, then uses symmetric INT8
  round-half-away-from-zero.
- The incomplete tail is stored in a separate BF16 tail cache
  `[pools, kpool, 128]`. Main KV pages follow the configured attention block
  size and allocator-provided block stride.
- For the GLM-5.3 sample the top-k buffer width is `round_up(2048 + 4 - 1,
  128) == 2176`; entries are `int32`. Decode retains `kpool-1` tail columns
  after the logical top-k history.
- Compressed block tables, physical slot IDs and cache positions use stored
  pool units while main KV uses token units; expansion happens only at the
  documented attention boundary.

## 6. Other execution contracts

- Preserve KDA recurrent state, mHC post/pre fusion, multimodal tower
  projection, shared-expert routing, and MTP iteration behavior.
- MTP uses the same sparse top-k buffer contract and cache-position units.
- Sequence/context-parallel exchange reserves transient rows before top-k
  exchange and uses `cache_interleave_size / index_kpool` for pool units.

## 7. Chip scope and validation

- Bound sample: `GLM-5.3-Flash-dummy-3layers-w8a8`, TP1 dummy smoke.
- Source acceptance covers import audit, registration, K-pool INT8/BF16
  geometry and numerics, and top-k-buffer width.
- Final acceptance requires normal-entry prefill, two-token decode,
  concurrent batches, and `tools/batched_test --infer`.
- FP8/FP4 cache and compute paths remain unverified on C500.

## 8. Open obligations

- Final model e2e and batched evidence are recorded in validation.
- Revisit CUDA-only KDA/mHC and FP8/FP4 branches on explicitly supported
  hardware.
