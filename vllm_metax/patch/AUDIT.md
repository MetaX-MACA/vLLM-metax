# Runtime patch audit — 2026-09-17

Baseline: `/workspace/vllm`, commit `98dff2a81d`, matching the installed
`vllm==0.29.1.dev0+g98dff2a81.d20260917.empty`. Python: `/opt/venv/bin/python`;
Transformers 5.17.0; Torch 2.10.0+metax3.8.2.2; Triton 3.6.0+metax3.8.2.2.
This audit targets this installed upstream revision, not an unpinned remote main.

All active patch modules, registration mutations, and the disabled rejection
sampler were inspected. `utils.py` is installation infrastructure; `template/`
contains inactive examples. Package initializers only load the entries below.

| Module | Decision and upstream evidence |
| --- | --- |
| `bugfix/bytes_to_unicode.py` | Keep remote-tokenizer compatibility: Transformers 5.17 removed the GPT-2 export. Install only when missing. |
| `bugfix/tokenizer.py` | Update: preserve tokenizer.json through upstream's full-backend conversion branch. Remove the global Llama constructor replacement and forced ByteLevel decoder, which break native SentencePiece tokenizers. |
| `bugfix/draft_config_overrides.py` | Replace the copied `SpeculativeConfig.__post_init__` with the new `compose_draft_hf_overrides` hook. Keep MetaX dictionary inheritance; delegate callable/empty overrides to upstream. This preserves new model detection, Medusa handling, validation and `model_weights` handling. |
| Nomic patch formerly in `draft_config_overrides.py` | Remove. Upstream `117afeea46` / PR #41277 deliberately removed the old maximum-length reset when fixing dynamic NTK scaling; the patch reintroduced that obsolete behavior. |
| `bugfix/deepseek_v4/parallel_state.py` | Keep: upstream still enables PP send/recv all-gather based on divisibility; no MACA trap workaround is present. Signature unchanged. Distributed hardware validation remains necessary before removal. |
| `bugfix/int8_w8a8/int8_moe_config.py` | Keep only W8A8: upstream W8A16 already forwards SwiGLU parameters, but W8A8 still does not. W8A8 implementation otherwise matches upstream. |
| `bugfix/int8_w8a8/mla_chunked_prefill.py` | Replace the whole-method patch with `_get_kv_b_proj_input_dtype`. Upstream now shares this helper across context paths, but still returns INT8 for INT8 weights. Preserve its INT32, FP8, ModelOpt and packed-weight handling. |
| `bugfix/qwen3_5_moe_loading.py` | Keep: upstream AutoWeightsLoader and Qwen3.5 MTP do not dequantize ignored shared-expert gates from affected INT8 checkpoints. Wrapper signature remains compatible. |
| `bugfix/telechat3/telechat3_yarn.py` | Update: normalize a temporary config to YaRN before upstream length validation. Preserve the model's actual rope type; reject invalid explicit lengths through upstream validation instead of silently clamping afterward. |
| `bugfix/triton_support/rejection_sampler.py` | Delete (previously disabled). Upstream recovery sampling already tiles the vocabulary and additionally masks invalid vocabulary entries and resolves ties safely. |
| `bugfix/triton_support/kda.py` | Keep: body matches upstream, but its non-AMD autotune list still includes 32 warps unsupported by MACA. |
| `bugfix/triton_support/chunk_delta_h.py` | Keep: body and heuristics match upstream. Upstream selects 2–4 pipeline stages; MACA still needs one stage to fit shared memory. |
| `bugfix/minimax_m3/index_topk.py` | Keep: current score/decode APIs match. Upstream launches still need the MACA single-stage and minimum MMA tile adjustments. |
| `bugfix/minimax_m3/sparse_attn.py` | Keep: upstream still loads 128-token tiles. Preserve its current FP8 scales, cache strides, causal mask, PDL and split-K logic with MACA 16-token sub-tiles. |
| `bugfix/minimax_m3/load_weights.py` | Keep: upstream still unconditionally renames `weight_scale_inv` to `weight_scale`; MetaX block-FP8 exposes `weight_scale_inv`. Remaining loading logic matches. |
| `enhancement/device_allocator.py` | Delete both patches. `_patch()` already redirects `vllm.device_allocator.cumem`; upstream allocator selection and Worker.shutdown use this module on CUDA-like platforms. Removing the old shutdown copy restores elastic-EP cleanup. |
| `enhancement/MRV2/` | Delete empty registration namespaces; no implementation or effect. |
| `enhancement/chores/bench_serve_args.py` | Keep deterministic-temperature policy via a small wrapper around upstream argument registration; remove the 400-line copy. Both import aliases remain patched. |
| `enhancement/chores/maca_visible_device.py` | Keep: upstream worker only installs the supplied environment, and MetaX's platform control variable remains CUDA_VISIBLE_DEVICES. Mirror it only when supplied, preserve explicit MACA settings, and avoid mutating RPC input. |
| `enhancement/chores/remove_error_log.py` | Update to accept upstream `requires_softcap`, `kv_cache_block_size`, and `supports_fa4_hd256` parameters while selecting MetaX FA2. |
| `enhancement/distributed/cuda_wrapper.py` | Keep: upstream CUDA runtime binding lacks MACA symbols and its IPC-handle ABI differs. All current wrapper methods are present in MetaX. |
| `enhancement/distributed/pynccl_wrapper.py` | Keep: upstream NCCL symbol names are not MCCL symbols. Both library aliases remain necessary; current method signatures match. |
| `enhancement/distributed/utils_patch.py` | Keep MCCL and MX-SMI discovery; no equivalent upstream dispatch is present. |
| `enhancement/dbo.py` | Keep: upstream SM control still asserts CUDA or ROCm instead of accepting CUDA-like OOT platforms. Other initialization matches. |
| `enhancement/joyai_support/joyai_transformer_config.py` | Keep registry entry: upstream still omits `joyai_llm_flash` from MLA detection. Base converter behavior and registry contract match. |
| `enhancement/quant_kernels/fp8_block_kernel.py` | Keep: upstream `register_linear_kernel` still has no `fp8_block` type. Append the OOT entry without destroying CUDA/ROCm/XPU registries. |
| `enhancement/utils.py` | Keep INT8 kernel selection and W4A8 property used by MetaX experts. Match upstream per-token semantics by ignoring a supplied static scale. |
| `performance/grouped_topk_router.py` | Keep MetaX fused dispatch. Current routing signature, fallback scoring, correction bias, scaling, batch invariance, and ROCm handling match upstream outside the marked dispatch changes. |
| `performance/speculative_decode_perf.py` | Update to upstream's four regions: decode, short extend, long extend, first prefill. Only completed-prefill decodes are sorted by query length; delegate the ordinary path to upstream's method. |
