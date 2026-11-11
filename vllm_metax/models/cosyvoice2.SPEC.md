# CosyVoice2

## 1. Identity and scope

- Registered architecture: `CosyVoice2ForCausalLM`.
- Target vLLM `v0.33.0`,
  `4ba787ec9959e566d3b587a796dff07673324da5`; prior vLLM `v0.32.0`.
- Complete causal/speech implementation is local in `cosyvoice2.py`.

## 2. Difference matrix

| Contract | Upstream CUDA | MACA C500 | Local implementation |
| --- | --- | --- | --- |
| Qwen decoder layers | Upstream concrete layer stack | Local copied Qwen layer API | `model_executor/models/qwen2.py` |
| Speech projection/head | Target CUDA dtype/projection | Local BF16 execution | `cosyvoice2.py` |

## 3. Local model dependencies

- The local Qwen2 layer implementation is owned by the legacy model-private
  dependency task.
- No upstream concrete Qwen model is inherited or constructed; import audit
  reports zero prohibited and zero unresolved dynamic imports.

## 4. Weights and compute

- Preserve target Qwen parameter names, speech projection/head ownership,
  dtype, and checkpoint mapping.
- C500 compute is BF16. Adapter/modality branches without a sample remain
  unverified.

## 5. Attention and cache

- Text self-attention uses the standard framework KV cache with model-derived
  head count/head size and runtime block size.
- Speech projections do not create a second attention cache. Cache insertion
  and reads use the same token slot IDs as the decoder.
- No custom MTP, DSpark, compressed or SWA cache is created by this model.

## 6. Other execution contracts

- Preserve embedding, normalization, MLP, speech feature projection, and
  output-head ownership.
- Preserve multimodal/audio preprocessing contracts from the target model.

## 7. Chip scope and validation

- No CosyVoice sample exists under `tools/models/`.
- Source acceptance covers import audit, registration, and model-free plugin
  closure.

## 8. Open obligations

- Add a supported sample before claiming model-level speech generation.
