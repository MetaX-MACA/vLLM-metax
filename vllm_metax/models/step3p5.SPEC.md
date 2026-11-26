# Step-3.5 model

## 1. Identity and scope

- Registered architectures: `Step3p5ForCausalLM`, `Step3p5MTP`.
- Target vLLM `v0.34.0`,
  `62d4bc3c7e33b1b6c906e233e6af2d3506c482ca`; prior vLLM `v0.33.0` (
  `4ba787ec9959e566d3b587a796dff07673324da5`).
- Complete model and MTP are local in `step3p5.py` and `step3p5_mtp.py`.

## 2. Difference matrix

| Contract | Upstream CUDA | MACA C500 | Local implementation |
| --- | --- | --- | --- |
| Router GEMM | CUTLASS `(352,4096)` shape tiers | Source native router kernel with retained MetaX shapes | `csrc/.../fp32_router_gemm*` |
| MoE | CUDA backends | Ordinary MetaX MoE; MegaMoE rejects | registry/MoE dispatch |
| SwiGLU clamp | Target CUDA | Shared portable activation path | local Step model |

## 3. Local model dependencies

- Uses local `step3p5.py`, `step3p5_mtp.py`, and public framework APIs.

## 4. Weights and compute

- Preserve target router shape `(352,4096)`, shared-expert routing, SwiGLU
  clamp, sharding, and MTP mappings.
- Use BF16 and ordinary MoE on C500. MegaMoE and packed FP8/FP4 paths
  require their corresponding kernel and quantization backends.

## 5. Attention and cache

- Uses standard framework attention/KV cache with runtime block size and
  model-provided head count/head size.
- Cache insertion and attention reads share token slot IDs in prefill, decode,
  mixed batches, and MTP.
- No custom quantized KV, compressed or sparse cache is created.

## 6. Other execution contracts

- Preserve dense/MoE layer selection, shared-expert accumulation, tensor and
  pipeline sharding, and exactly-once output reductions.
- MTP preserves target draft-layer indexing and the parent hidden-state source.
  The vLLM loader owns missing-parameter checks and exempts optional KV scales.

## 7. MACA porting recipe

Copy the target Step config resolvers into `step3p5.py` and import them from
`step3p5_mtp.py`. Form the absolute layer list from `layer_types` and
`mtp_layer_types`, and the expert list from `mlp_layer_types` and
`mtp_mlp_layer_types`. Preserve the remote-code variants for sliding heads,
layer-specific RoPE and `moe_layers_enum`. Use the target shared MTP loader
completeness checks. Keep the local router and expert-output reduction at the
same points in the target decoder flow.
