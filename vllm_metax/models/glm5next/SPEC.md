# GLM-5 Next MetaX specification

## 1. Identity and scope

- Registered architectures: `Glm5NextForCausalLM`,
  `Glm5NextForConditionalGeneration`, and `Glm5NextMTPModel`.
- Target vLLM `v0.34.0`,
  `62d4bc3c7e33b1b6c906e233e6af2d3506c482ca`; prior vLLM `v0.33.0` (
  `4ba787ec9959e566d3b587a796dff07673324da5`).
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
- The model uses local GLM4.1V vision and local DeepSeek MoE/draft helpers.

## 4. Weights and compute

- Sample `GLM-5.3-Flash-dummy-3layers-w8a8` has three text layers, hidden size
  4096, 64 Q/KV heads, QK/V dimensions 256, 288 routed experts, eight active
  experts, one shared expert, `index_topk=2048`, and `index_kpool=4`.
- Runtime activation/weight quantization is compressed-tensors W8A8.
- Preserve fused routing, shared experts, KDA convolution/state mappings,
  multimodal projection, checkpoint names, and post-load repacking.
- The weight loader preserves checkpoint INT8 scales when an attention
  projection stores INT8 weights.
- Use BF16/INT8 on C500. An FP8/FP4 branch requires a matching kernel
  and support for its cache and activation formats.

## 5. Attention and indexer cache

- Sparse K-pool pages store all 128-byte INT8 quantized rows first and one
  FP32 scale per row after the value region. Rows are their pool slots in
  stored-state units.
- A pool is `index_kpool` consecutive token keys. The INT8 path applies
  Hadamard-128 rotation, materializes BF16, then uses symmetric INT8
  round-half-away-from-zero.
- Local top-k consumes FP32. When BF16 logits are requested, scores
  are rounded to BF16 before conversion to the native top-k input. Scaled
  MXFP4 indexer queries require a separate FP8/FP4 backend.
- The incomplete tail is stored in a BF16 cache
  `[num_blocks, 2, ring_size, 128]`, with keys and gate scores in separate
  planes. The ring holds at least one pool plus the speculative tokens,
  rounded up to a power-of-two number of pools. A token's tail slot is
  `block * ring_size + pos % ring_size`. Pool completion still uses
  `pos % index_kpool`. Main KV pages use the allocator's block stride.
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

## 7. MACA porting recipe

Port the target KDA merged QKV convolution and parameter shard loader
together. Keep the MACA Triton prefill path and its L2 normalization warmup.
Use the local GLM 4.1V vision helpers and the local attention selector. For
pre-sampled video, use the supplied frame indices for timestamps.

Allocate the tail ring from the pool size and speculative width. Pass the same
ring size to the BF16 and INT8 writers. Use logical position modulo ring size
for physical history lookup and modulo pool size for compression. In graph
decode put padded tokens in a disposable request row. In ShardQ prefill gather
Q, scales, weights and positions with the same token indices. Round BF16
logits at the score stage, then convert to FP32 for MACA top-k.

Reuse loaded INT8 weights and preserve their checkpoint scales. Build/refill
the FP32 head-gate derived tensor after loading before the indexer consumes
it. Keep MTP registration and expert-prefix selection aligned with the parent.
Use the regular MACA W8A8 experts on C500. The upstream MegaMoE branch requires
a compatible backend, EP, TP=1 and its own block-scale loader.
