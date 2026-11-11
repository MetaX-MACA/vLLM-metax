# DeepSeek V2/V3, V3.2 and GLM MoE-DSA

## 1. Identity and scope

- Registered architectures: `DeepseekV2ForCausalLM`,
  `DeepseekV3ForCausalLM`, `DeepseekV32ForCausalLM`,
  `GlmMoeDsaForCausalLM`.
- Target vLLM `v0.33.0`,
  `4ba787ec9959e566d3b587a796dff07673324da5`; prior vLLM `v0.32.0`.
- Implementation is the complete local `deepseek_v2.py`.

## 2. Difference matrix

| Contract | Upstream CUDA | MACA C500 | Local implementation |
| --- | --- | --- | --- |
| Sparse indexer | CUDA FP8 index cache/top-k | BF16/INT8 local cache and top-k | local indexer and registry dispatch |
| MLA | CUDA sparse MLA | Local MLA backends | local `deepseek_v2.py` with framework MLA |
| MoE | Target CUDA methods | MetaX fused MoE dispatch | registry/MoE contracts |
| Draft/index sharing | Target semantics | Explicit local indexer lifetime | local decoder/MTP |

No upstream concrete model inheritance or construction is permitted.

## 3. Local model dependencies

- The local flat model imports public framework APIs and the local
  `vllm_metax.model_executor.models` Qwen/DeepSeek helpers.
- Import audit reports no prohibited concrete implementation or unresolved
  dynamic import.

## 4. Weights and compute

- Sample `GLM-5_2-W8A8-dummy-3layers` has hidden size 6144, three layers,
  64 Q/KV heads, QK 192 NoPE + 64 RoPE, V head 256, 256 routed experts, eight
  active experts, one shared expert, and `index_topk=2048`.
- Preserve target MLA, sparse indexer, DSA top-k, MoE/shared-expert mappings,
  PP/EP/TP ownership, and checkpoint loading.
- C500 uses BF16/INT8; FP8/FP4 branches remain unverified.

## 5. Attention and indexer cache

- MLA K/V uses the runtime attention block size and config-derived 576
  semantic head dimensions where selected. Cache layouts remain framework
  `MLAAttentionSpec` layouts with allocator strides.
- The indexer cache is 132 bytes per stored state: 128 BF16/INT8/FP8 payload
  bytes according to the selected implementation plus one FP32 scale per
  128 values.
- `index_topk=2048` and `index_head_dim=128` for the bound sample. The top-k
  buffer retains `int32` token IDs and is reused by MTP when index sharing is
  enabled.
- `index_topk_pattern="FFS"` and `index_topk_freq=4` determine which layers
  own an indexer. MTP never starts with a skipped/uninitialized top-k buffer.

## 6. Other execution contracts

- Preserve grouped top-k correction bias, shared experts, MLA inverse
  projection, and pipeline ownership.
- MTP index sharing reuses the parent indexer only after the target iteration
  state is available; it must not read an unwritten buffer.

## 7. Chip scope and validation

- Bound sample: `GLM-5_2-W8A8-dummy-3layers`, TP1 dummy smoke.
- Source acceptance covers import audit, registration, indexer/top-k geometry
  and local dispatch.
- Final acceptance requires normal-entry prefill, two-token decode,
  concurrent batches, and `tools/batched_test --infer`.

## 8. Open obligations

- Final model evidence is recorded in validation.
- Validate FP8/FP4 indexer/cache variants only on supported hardware.
