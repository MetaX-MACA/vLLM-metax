# MiniMax M3 DSpark compatibility entry

## 1. Identity and scope

- Registered architecture: `DSparkMiniMaxDraftModel`.
- Target vLLM `v0.34.0`,
  `62d4bc3c7e33b1b6c906e233e6af2d3506c482ca`; prior vLLM `v0.33.0`.
- `minimax_m3_dspark.py` is a compatibility registration surface for the
  complete local `minimax_m3` implementation.

## 2. Difference matrix

| Contract | Upstream CUDA | MACA C500 | Local implementation |
| --- | --- | --- | --- |
| Draft model construction | NVIDIA DFlash lifecycle | Local MiniMax draft lifecycle | `minimax_m3/metax/dspark.py` |
| Sparse/cache selection | CUDA MSA | Local Triton/BF16 selection | `minimax_m3` local modules |

## 3. Local model dependencies

- Delegates only to the local `vllm_metax.models.minimax_m3` package.
- `metax/dspark.py` uses local MiniMax attention and derives its draft
  lifecycle from the local Qwen3 DSpark and DFlash modules.

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

## 7. MACA porting recipe

Resolve `DSparkMiniMaxDraftModel` to the local MiniMax M3 draft. Match the
target weight aliases and context layout in `common/dspark.py`.
Reapply the MACA projection and Gemma K norm in `metax/dspark.py`.
Preserve the parent model's auxiliary hidden-state order and draft KV
positions when rebuilding the local draft. The chosen attention and quantization contracts come from MiniMax
M3, so apply that model's MACA backend gate to the draft as well.
