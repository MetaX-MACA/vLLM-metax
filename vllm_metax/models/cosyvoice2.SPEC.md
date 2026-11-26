# CosyVoice2

## 1. Identity and scope

- Registered architecture: `CosyVoice2ForCausalLM`.
- Target vLLM `v0.34.0`,
  `62d4bc3c7e33b1b6c906e233e6af2d3506c482ca`; prior vLLM `v0.33.0` (
  `4ba787ec9959e566d3b587a796dff07673324da5`).
- Complete causal/speech implementation is local in `cosyvoice2.py`.

## 2. Difference matrix

| Contract | Upstream CUDA | MACA C500 | Local implementation |
| --- | --- | --- | --- |
| Qwen decoder layers | Upstream concrete layer stack | Local copied Qwen layer API | `vllm_metax/models/qwen2.py` |
| Speech projection/head | Target CUDA dtype/projection | Local BF16 execution | `cosyvoice2.py` |

## 3. Local model dependencies

- Construct the decoder from the local `qwen2.py` implementation and use
  the shared framework interfaces for the LM head and pipeline boundary.

## 4. Weights and compute

- Preserve target Qwen parameter names, speech projection/head ownership,
  dtype, and checkpoint mapping.
- The C500 speech-token decoder uses BF16 compute.

## 5. Attention and cache

- Text self-attention uses the standard framework KV cache with model-derived
  head count/head size and runtime block size.
- Speech projections do not create a second attention cache. Cache insertion
  and reads use the same token slot IDs as the decoder.
- No custom MTP, DSpark, compressed or SWA cache is created by this model.

## 6. Other execution contracts

- Preserve embedding, normalization, MLP, speech feature projection, and
  output-head ownership.
- The model generates speech tokens and uses the local Qwen2 text decoder.

## 7. MACA porting recipe

Use the target legacy CosyVoice2 model as the baseline and replace its concrete
`Qwen2Model` import with the local `qwen2.py` mirror. Keep the shared loader,
tensor-parallel LM head and logits processor imports from vLLM. Preserve the
speech-token vocabulary and the decoder-to-head path. Adapt attention in the
local Qwen2 implementation through MACA backends, without inserting a separate
speech attention or cache operator.
