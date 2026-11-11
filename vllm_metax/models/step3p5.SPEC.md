# Step-3.5 model

## 1. Identity and scope

- Registered architectures: `Step3p5ForCausalLM`, `Step3p5MTP`.
- Target vLLM `v0.33.0`,
  `4ba787ec9959e566d3b587a796dff07673324da5`; prior vLLM `v0.32.0`.
- Complete model and MTP are local in `step3p5.py` and `step3p5_mtp.py`.

## 2. Difference matrix

| Contract | Upstream CUDA | MACA C500 | Local implementation |
| --- | --- | --- | --- |
| Router GEMM | CUTLASS `(352,4096)` shape tiers | Source native router kernel with retained MetaX shapes | `csrc/.../fp32_router_gemm*` |
| MoE | CUDA backends | Ordinary MetaX MoE; MegaMoE rejects | registry/MoE dispatch |
| SwiGLU clamp | Target CUDA | Shared portable activation path | local Step model |

## 3. Local model dependencies

- Uses local `step3p5.py`, `step3p5_mtp.py`, and public framework APIs.
- No upstream concrete Step model is inherited or constructed; import audit
  reports zero prohibited and zero unresolved dynamic imports.

## 4. Weights and compute

- Preserve target router shape `(352,4096)`, shared-expert routing, SwiGLU
  clamp, sharding, and MTP mappings.
- C500 support is BF16 with ordinary MoE. FP8/FP4 and unsupported MegaMoE
  paths reject or remain unverified.

## 5. Attention and cache

- Uses standard framework attention/KV cache with runtime block size and
  model-provided head count/head size.
- Cache insertion and attention reads share token slot IDs in prefill, decode,
  mixed batches, and MTP.
- No custom quantized KV, compressed or sparse cache is created.

## 6. Other execution contracts

- Preserve dense/MoE layer selection, shared-expert accumulation, tensor and
  pipeline sharding, and exactly-once output reductions.
- MTP preserves target draft-layer indexing and parent hidden-state source.

## 7. Chip scope and validation

- No Step model sample exists under `tools/models/`.
- Source acceptance covers import audit, registration, and router shape/numeric
  compatibility.

## 8. Open obligations

- Add and bind a Step dummy sample before claiming model-level validation.
