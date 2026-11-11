# MiniMax M3 DSpark compatibility entry

## 1. Identity and scope

- Registered architecture: `DSparkMiniMaxDraftModel`.
- Target vLLM `v0.33.0`,
  `4ba787ec9959e566d3b587a796dff07673324da5`; prior vLLM `v0.32.0`.
- `minimax_m3_dspark.py` is a compatibility registration surface for the
  complete local `minimax_m3` implementation.

## 2. Difference matrix

| Contract | Upstream CUDA | MACA C500 | Local implementation |
| --- | --- | --- | --- |
| Draft model construction | NVIDIA DFlash lifecycle | Local MiniMax draft lifecycle | `minimax_m3/metax/dspark.py` |
| Sparse/cache selection | CUDA MSA | Local Triton/BF16 selection | `minimax_m3` local modules |

## 3. Local model dependencies

- Delegates only to the local `vllm_metax.models.minimax_m3` package.
- It does not import or inherit an upstream concrete MiniMax model; import
  audit reports zero prohibited and zero unresolved dynamic imports.

## 4. Weights and compute

- Draft context projection, parent hidden-state ownership, and load mappings
  follow `minimax_m3/SPEC.md`.
- C500 compute is BF16 with supported INT8 experts; NVIDIA NVFP4/MSA is not
  selected.

## 5. Attention and cache

- Uses the parent MiniMax main KV and key-only index caches.
- Draft cache positions, speculative rows, and selected physical blocks follow
  the same units and lifetime as `minimax_m3`.
- Rejected draft rows are overwritten or ignored by the parent contract, not
  interpreted as committed index keys.

## 6. Other execution contracts

- Preserve parent/self-speculative draft sharing, context projection, logits
  ownership, and MTP iteration state.

## 7. Chip scope and validation

- No MiniMax sample is bound; model-level DSpark validation is pending.
- Source acceptance covers import audit, registration, and local parent
  dispatch.

## 8. Open obligations

- Validate DSpark with a bound MiniMax dummy sample after the parent source and
  patch phases are complete.
