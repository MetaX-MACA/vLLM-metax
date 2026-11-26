# DeepSeek V2/V3, V3.2 and GLM MoE-DSA

## 1. Identity and scope

- Registered architectures: `DeepseekV2ForCausalLM`,
  `DeepseekV3ForCausalLM`, `DeepseekV32ForCausalLM`,
  `GlmMoeDsaForCausalLM`.
- Target vLLM `v0.34.0`,
  `62d4bc3c7e33b1b6c906e233e6af2d3506c482ca`; prior vLLM `v0.33.0` (
  `4ba787ec9959e566d3b587a796dff07673324da5`).
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

- The flat model owns the DSA indexer and decoder classes. It uses the shared
  vLLM layer, loader and interface APIs and the MetaX MLA/top-k layers.

## 4. Weights and compute

- Sample `GLM-5_2-W8A8-dummy-3layers` has hidden size 6144, three layers,
  64 Q/KV heads, QK 192 NoPE + 64 RoPE, V head 256, 256 routed experts, eight
  active experts, one shared expert, and `index_topk=2048`.
- Preserve target MLA, sparse indexer, DSA top-k, MoE/shared-expert mappings,
  PP/EP/TP ownership, and checkpoint loading.
- Use BF16/INT8 on C500. Packed low-precision paths require a backend that
  supports the selected cache dtype and the model's weight format.

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

## 7. MACA porting recipe

Port the target DeepSeek V2/V3/V3.2 loader and retain the local DSA/MLA
implementation used by the `GlmMoeDsaForCausalLM` alias. Replace the single
shared-expert flag with target `get_fused_shared_expert_layers` and account for
the EPLB setting when building expert mappings. Keep layer-local routing and
the local MLA and INT8 indexer selection. For W8A8, pass the
model activation dtype to KV-B projection so the MACA linear method performs
dynamic activation quantization.
