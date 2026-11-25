# MetaX model registry

- Upstream baseline: vLLM `62d4bc3c7e33b1b6c906e233e6af2d3506c482ca` (v0.34.0).
- Prior baseline: MetaX `c67bedb8` on `master` (vLLM v0.33.0).
- Registry entry: `vllm_metax/models/__init__.py`.

The registry keeps vLLM's `ModelRegistry.register_model` extension point and resolves
every registered architecture through a local `vllm_metax.models` implementation.
Architectures and their local owners:

| Architecture | Local implementation |
| --- | --- |
| CosyVoice2ForCausalLM | `vllm_metax.models.cosyvoice2` |
| DeepSeekMTPModel / DeepseekV32MTPModel | `vllm_metax.models.deepseek_mtp` |
| DeepseekV2ForCausalLM / DeepseekV3ForCausalLM / DeepseekV32ForCausalLM / GlmMoeDsaForCausalLM | `vllm_metax.models.deepseek_v2` |
| DeepseekV4ForCausalLM / DeepSeekV4MTPModel / DSparkDraftModel | `vllm_metax.models.deepseek_v4` |
| DeepseekV41ForCausalLM / DSparkDeepseekV4ForCausalLM / DSparkV41DraftModel | `vllm_metax.models.deepseek_v41` |
| Step3p5ForCausalLM / Step3p5MTP | `vllm_metax.models.step3p5`, `vllm_metax.models.step3p5_mtp` |
| Qwen4ExpForCausalLM / Qwen4ExpForConditionalGeneration / Qwen4ExpMTP | `vllm_metax.models.qwen4_exp` |
| Glm5NextForCausalLM / Glm5NextForConditionalGeneration / Glm5NextMTPModel | `vllm_metax.models.glm5next` |
| MiniMaxM3SparseForCausalLM / MiniMaxM3SparseForConditionalGeneration / MiniMaxM3MTP / DSparkMiniMaxDraftModel | `vllm_metax.models.minimax_m3` |

The v0.34 registry API and vLLM model construction contracts are preserved. MetaX
adds only local architecture/provider mappings; the registry must not delegate a
registered architecture to an upstream concrete model implementation. The complete
local model mirrors are owned by their corresponding `model-*` tasks and must be
reviewed before this registry is considered source-ready.
