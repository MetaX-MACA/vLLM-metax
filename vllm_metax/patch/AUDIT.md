# Runtime patch audit — vLLM 0.32.0

Reviewed upstream target: **v0.32.0**
(`5df63a7203cdb9328d824a27b682e77c7b9928be`). Patch paths below are relative
to `vllm_metax/patch/`; upstream source paths are relative to `vllm/`.
Transformers compatibility entries refer to the approved **5.19.0** environment.

This inventory describes the current working tree: **32 patch modules**
(12 bugfix, 5 model-specific and 15 enhancement, including 3 performance),
plus shared infrastructure.
Decisions describe the retained implementation or its migration/removal;
**they do not declare all v0.32.0 runtime obligations complete**. Earlier 0.30.0
checks and patch-header dates remain historical evidence, not current acceptance.
Environment approval, source fingerprints and individual check results are maintained
in the [upgrade run](../../.metax-upgrade/setup-preview/TODO.md).

## Activation

`vllm_metax:register_out_of_tree` calls `_ensure_global_patch()` before
`_out_of_tree()`. `register_attn_backends` uses the same once-per-process guard.
The guard imports `bugfix`, `enhancement`, then `models`; their package
initializers import the modules listed below. `enhancement` imports its
`performance` subpackage last, after allocator/communication/kernel setup.
`_out_of_tree()` then loads the ordinary registries used by migrated implementations. Import activation is distinct
from executing a patched branch or selecting its native kernel.

Entries in the four active tables are imported through this chain.
`enhancement/performance/gumbel_sample.py` installs its replacement only when the
optional MCOPLIB sampler is available. Removed/migrated entries have a separate table;
templates are not imported by the activation chain.

## Bugfix patches

| Module / target | Decision, repair reason and removal condition |
| --- | --- |
| `bugfix/bytes_to_unicode.py`: `transformers.models.gpt2.tokenization_gpt2.bytes_to_unicode` | **Retain.** Restore the cached GPT-2 byte mapping only when the export is absent; affected remote tokenizers still import that legacy Transformers location. Remove when those tokenizers no longer require the removed export. Remote-tokenizer loading remains a separate validation obligation. |
| `bugfix/tokenizer.py`: `LlamaTokenizerFast.convert_to_native_format`, `AutoTokenizer.from_pretrained` | **Retain.** Select full serialized-backend loading when a real `tokenizer.json` is supplied; otherwise delegate native conversion. After normal loading of a local JoyAI checkpoint, restore its serialized backend while retaining added-token flags and checking token IDs. Remove each branch when Transformers/checkpoints preserve the same backend semantics without it. Llama and JoyAI round-trip behavior require separate checks. |
| `bugfix/draft_config_overrides.py`: `SpeculativeConfig.compose_draft_hf_overrides`, `SpeculativeConfig.hf_config_override` | **Retain.** Target `config/speculative.py` intentionally inherits callable overrides but leaves dictionaries target-only. MetaX composes picklable dictionary/nested-config updates and preserves JoyAI's `joyai_dense_mtp` identity before the upstream draft rewrite; MiniMax DSpark also keeps its standalone architecture, flattened text config and independent quantization before model construction; other cases delegate. Empty/None overrides also use the callable object: the patched static-method module export is a descriptor, so returning its underlying function fails standard pickle. Remove when supported upstream configuration hooks cover these requirements. Verify nested overrides, serialization and actual draft selection. |
| `bugfix/minimax_m3_draft_config.py`: `SpeculativeConfig.__post_init__` | **Retain configuration-only bridge; use an explicit MiniMax DSpark initializer.** Target v0.32 selects the target checkpoint for implicit DSpark and rewrites unrecognized architectures to DeepSeek before registry inspection and parallel-config construction. For MiniMax text/VL targets with `method='dspark'`, resolve local `dspark/config.json` or use the explicit draft path, require `DSparkMiniMaxDraftModel`, and retain independent draft quantization. Reuse upstream TP, maximum-length and parallel-config helpers, preserve applicable DSpark checks and Pydantic validation. All other targets/methods delegate to the original initializer. No AST rewriting or `exec` remains. This cannot live in a model constructor because configuration selection runs first. Remove when upstream supplies both config behaviors. |
| `bugfix/parallel_state.py`: `GroupCoordinator._should_use_all_gather` | **Retain.** Disable the PP send/recv all-gather optimization for the recorded MACA trap; target `distributed/parallel_state.py` still provides this optimization. Remove after MACA PP/TP validation demonstrates the native path is safe. This is a hardware workaround, not a general prohibition on all-gather. |
| `bugfix/int8_w8a8/int8_moe_config.py`: `int8_w8a8_moe_quant_config` | **Retain.** Target `model_executor/layers/fused_moe/config.py` still omits `gemm1_alpha`, `gemm1_beta` and `gemm1_clamp_limit` from the W8A8 constructor. Forward these SwiGLU parameters for affected models. Remove when the upstream W8A8 constructor forwards all three fields. |
| `bugfix/int8_w8a8/mla_chunked_prefill.py`: `_get_kv_b_proj_input_dtype` | **Retain.** For INT8 weights, preserve model-dtype activations because the MetaX linear method dynamically quantizes them. Delegate all other cases to target `model_executor/layers/attention/mla_attention.py`, including packed UINT8, INT32, ModelOpt and FP8 behavior. Remove when upstream supports this MetaX activation contract. |
| `bugfix/triton_support/rejection_sample.py`: `_insert_resampled_kernel`, `rejection_sample` | **Migrate insert; retain launch workaround.** Move the unchanged MC3-15766 four-clamp insert kernel from `bugfix/rejection_sampler.py`, retire its old activation entry, and copy the exact target `v1/worker/gpu/spec_decode/rejection_sampler_utils.py:rejection_sample` with only the resample block size changed from 1024 to 512. Preserve watermark deduplication, block verification and draft-stride contracts. The provider and preloaded caller aliases bind to the replacements. Remove the clamps after equivalent upstream bounds or a verified MACA fix; remove the launch replacement after reliable 1024-token MACA execution or an upstream tuning hook. CPU launch capture and source equality do not establish native GPU correctness for empty, padded or changing batches. |
| `bugfix/routed_experts_fused_weight_fix.py`: `RoutedExperts.load_weights` | **Retain.** For transposed fused checkpoints, normalize only CHANNEL scale tensors whose singleton axis is in the middle, e.g. `[E, 1, N]` to `[E, N, 1]`. Resolve the destination parameter before normalization; leave BLOCK scales unchanged and delegate the complete upstream loader. Remove when upstream or re-quantized checkpoints provide the native per-channel layout. Verify gate/up splitting, down projection and reloads. |
| `bugfix/triton_support/kda.py`: `kda_gate_fwd_kernel` | **Update; retain MACA tuning.** Refresh the target masked-pointer implementation from `third_party/flash_linear_attention/ops/kda.py`, retaining the autotune restriction that excludes 32 warps. Remove when native tuning selects configurations supported on MACA. Body/launch contract checks do not establish native numerical correctness. |
| `bugfix/triton_support/chunk_delta_h.py`: `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | **Update; retain MACA tuning.** Refresh target masked-pointer behavior from `third_party/flash_linear_attention/ops/chunk_delta_h.py` and retain single-stage tuning for the recorded shared-memory limit. Remove when upstream tuning or the backend resolves that restriction. Variable-length tails and native numerics remain validation obligations. |
| `bugfix/triton_support/fused_moe_lora_shmem.py`: `get_max_shared_memory_bytes` | **Retain existing activation; inventory correction.** Target `lora/ops/triton_ops/fused_moe_lora_op.py` receives the regular per-block limit (64 KiB fallback) so the one-shot MoE-LoRA launch can choose its smaller pipeline. Remove when upstream uses the limit enforced by MACA Triton. Current LoRA numerics/resource behavior is outside this layout review. |

## Enhancement patches

| Module / target | Decision, repair reason and removal condition |
| --- | --- |
| `enhancement/device_allocator.py`: submodule `vllm.device_allocator.cumem` | **Migrate activation.** Move the plugin-entry `sys.modules` redirect into an explicit `patch(..., replace_module=True)` installation. Update both the import cache and package attribute without executing the original CUDA allocator. Rebind preloaded same-name allocator class/function imports; renamed imports and existing allocations are outside the migration. Target `device_allocator/__init__.py:get_mem_allocator_instance` and GPU worker shutdown lazily import this submodule. Remove when native platform allocator dispatch covers both consumers. Fresh startup/import identity checks do not establish allocator sleep/wake numerical or lifecycle correctness. |
| `enhancement/chores/bench_serve_args.py`: `add_cli_args` | **Retain policy.** Wrap target `benchmarks/serve.py` argument registration and default temperature to 0.0, preserving explicit values and the full argument set. Same-name loaded CLI aliases are handled by the installer. Remove when MetaX benchmark policy adopts the native default. This is a benchmark policy, not a correctness fix. |
| `enhancement/chores/maca_visible_device.py`: `WorkerWrapperBase.update_environment_variables` | **Retain.** Copy the selected RPC environment dictionary and mirror CUDA visibility to MACA visibility only when supplied; explicit MACA values win. Target `v1/worker/worker_base.py` installs the supplied variables without this translation. Remove when native worker launch provides equivalent MetaX propagation. |
| `enhancement/chores/remove_error_log.py`: `get_flash_attn_version` | **Retain.** Redirect target `v1/attention/backends/fa_utils.py` selection to MetaX FA2 and synchronize loaded same-name callers, including vision attention imports. The wrapper accepts the target's selection arguments and currently forwards the MetaX subset; feature-specific compatibility still needs validation. Remove when a supported native platform hook selects MetaX FA without NVIDIA probing. |
| `enhancement/distributed/cuda_wrapper.py`: `cudaIpcMemHandle_t`, `CudaRTLibrary` | **Retain.** Replace target `distributed/device_communicators/cuda_wrapper.py` exports with MACA runtime bindings and the 128-byte IPC handle. Same-name loaded class aliases must also resolve to the replacement. Remove when upstream provides equivalent MACA symbols and ABI. Native IPC/host-memory behavior is separate from successful alias installation. |
| `enhancement/distributed/pynccl_wrapper.py`: `NCCLLibrary` | **Retain.** Bind the defining `distributed/device_communicators/pynccl_wrapper.py` export to `MCCLLibrary`; the installer also synchronizes loaded `pynccl` aliases. Remove when upstream dispatches to MCCL symbols natively. Actual collectives require distributed validation. |
| `enhancement/distributed/utils_patch.py`: `find_nccl_library`, `import_pynvml` | **Retain.** Redirect target `utils/nccl.py` and `utils/import_utils.py` discovery to MCCL and MX-SMI. Synchronize previously loaded same-name bindings such as `pynccl_wrapper.find_nccl_library`. Remove when native discovery supports MetaX libraries. |
| `enhancement/dbo.py`: `SMControlContextManager.__init__` | **Retain.** Accept the CUDA-like MetaX platform in target `v1/worker/ubatch_utils.py`, evaluating `is_cuda_alike()` as a boolean while retaining SM resource checks and callbacks. Remove when upstream accepts this OOT platform through the native path. DBO overlap and resource allocation still require native execution checks. |
| `enhancement/quant_kernels/fp8_block_kernel.py`: `_POSSIBLE_FP8_BLOCK_KERNELS[PlatformEnum.OOT]` | **Retain private registration.** Target `model_executor/kernels/linear/__init__.py` does not expose a `"fp8_block"` branch in `register_linear_kernel`. Append the MetaX candidate only to OOT, preserving other platforms. Migrate when the public API accepts this kernel type. Actual kernel selection and numerical behavior require separate checks. |
| `enhancement/utils.py`: `_int8_quantize` | **Retain.** Use MetaX scaled INT8 quantization for the per-token branch in `model_executor/layers/fused_moe/utils.py`. Preserve per-tensor static/dynamic and block quantization behavior; supplied static scale is intentionally not used in dynamic per-token mode. Remove when native INT8 dispatch implements the same MetaX contract. |
| `enhancement/utils.py`: `FusedMoEQuantConfig.use_int4_w4a8` | **Retain intentional attribute.** Add the property used by MetaX W4A8 experts for string descriptors `"int8"`/`"int4"`. Its absence upstream is intentional compatibility scope, so this target uses `allow_missing=True`. Remove when upstream provides the descriptor or MetaX no longer uses it. |
| `enhancement/utils.py`: `moe_kernel_quantize_input` | **Retain narrow extension.** Intercept string `"int8"`, reject unsupported emulation and delegate every other dtype/argument to the captured target implementation. The installer synchronizes loaded same-name aliases. Remove when upstream handles the MetaX string descriptor and quantizer equivalently. |
| `enhancement/MRV2/model_runner.py`: `AsyncOutput`, `GPUModelRunner.__init__` | **Retain existing conditional activation; inventory correction.** When `VLLM_METAX_DISABLE_ASYNC_OUTPUT` is enabled, target `v1/worker/gpu/model_runner.py` uses synchronous output copying for MetaX debugging. Remove after a native copy-stream option or resolution of the recorded output-copy issue. MRV2 execution with the option is outside this layout review. |
| `enhancement/indexer_kv_dtype.py`: `IndexerKVDType`, `AttentionConfig.indexer_kv_dtype` | **Retain existing activation; inventory correction.** Extend target `config/attention.py` with INT8 and rebuild Pydantic metadata. Remove when the native config accepts INT8. This layout review does not validate INT8 cache execution. |

## Shared performance enhancements

| Module / target | Decision, repair reason and removal condition |
| --- | --- |
| `enhancement/performance/grouped_topk_router.py`: `GroupedTopKRouter._compute_routing` | **Update; retain MetaX dispatch.** Preserve target `model_executor/layers/fused_moe/router/grouped_topk_router.py` scoring/bias/scaling behavior around the MACA fused/SGL/fallback routes. SGL requires precompiled kernels: `USE_PRECOMPILED_KERNEL=0` with the SGL preference enabled uses native grouped-topk, since source mode does not load `mcoplib.sgl_kernel`. Apply target skip-padding semantics when enabled: padding rows receive zero weights and expert IDs -1. Remove when native router extension points support these MetaX kernels. Source-mode sigmoid routing, bias, scaling, patch binding and graph replay have scoped GPU component diagnostics in `../../.metax-upgrade/setup-preview/csrc-upgrade/`; formal whole-tree acceptance remains pending concurrent model edits. Precompiled SGL numerics and full MoE model execution remain separate obligations. |
| `enhancement/performance/gumbel_sample.py`: `Sampler._sample_random` | **Migrate; conditional activation.** Replace only the target random-sampling hook in `v1/worker/gpu/sample/sampler.py`, preserving `Sampler.sample` sequence-length handling and fused-sampler branches. MCOPLIB receives `all_greedy=True` only when all selected temperatures are zero. If `mcoplib.triton_gumbel_sample` cannot be imported, leave the upstream sampler installed. Remove when native sampling supports the equivalent MetaX kernel. The approved component snapshot lacks this optional MCOPLIB module; fallback execution is not optimized-sampler validation. |
| `enhancement/performance/speculative_decode_perf.py`: `GPUModelRunner._may_reorder_batch` | **Retain.** Delegate ordinary paths; when speculative decode/backend grouping is enabled, preserve the four request regions and stably bucket completed-prefill decodes by query length. Example: `[D:3, D:1, D:2, D:1, S:2, L:8, P:4]` becomes `[D:1, D:1, D:2, D:3, S:2, L:8, P:4]`. Keep request state aligned through swaps. Remove when native batch ordering provides the needed decode buckets; preserve each non-decode region's order without claiming upstream's subset sorting is globally stable. |

## Model patches

Model ownership takes priority over bugfix/enhancement/performance intent.
These packages are imported eagerly; model directory placement does not scope
a replacement automatically. Shared wrappers remain in the shared categories.

| Module / target | Decision, repair reason and removal condition |
| --- | --- |
| `models/deepseek_v4/topk_global_indices_oob.py`: `ComputeGlobalTopkIndicesAndLensKernel.kernel`, `ComputeGlobalTopkIndicesAndLensKernel.__call__` | **Retain.** Target `models/deepseek_v4/common/ops/cache_utils.py` supplies the declarative kernel/launcher being replaced. Pass current block-table bounds, validate request rows/block columns, and clamp pointer operands before gathering. Pack surviving slots into the prefix consumed by sparse attention: with block size 4, table `[7, 8, 9]` and local indices `[0, -1, 5, 12]`, produce `[28, 33, -1, -1]`, length 2. Preserve the registered kernel instance. Remove after upstream provides equivalent bounds and packed-prefix safety on MACA. Graph-padding, geometry changes and native-kernel validation remain required. |
| `models/joyai/joyai_transformer_config.py`: `MODEL_ARCH_CONFIG_CONVERTORS["joyai_llm_flash"]` | **Retain.** Add only the JoyAI converter entry in `transformers_utils/model_arch_config_convertor.py`; identify MLA from the checkpoint's compressed/non-compressed architecture fields and inherit other conversion behavior. Remove when upstream recognizes this architecture equivalently. |
| `models/kimi_k3/disable_low_latency.py`: `_low_latency_table` | **Retain; correct header evidence.** Return `None` from target `models/kimi_k3/nvidia/low_latency_gemm.py` so capability aliases cannot install NVIDIA projection plans or enable KDA projection overlap on MACA. Both `enable_kimi_k3_low_latency_gemm` (model/MTP weight loading) and `try_low_latency_gemm` (MLA) resolve this module global. Imports of CuTe kernels occur in later execution branches; eager-import failure was not reproduced. Remove when native dispatch checks backend support or equivalent MACA kernels are validated. Reviewed v0.32.0 only; the original v0.30.0 attribution had no evidence. Import/fallback checks do not establish Kimi-K3 model numerics. |
| `models/qwen/qwen3_5_moe_loading.py`: `AutoWeightsLoader.load_weights` | **Retain.** Dequantize ignored INT8 shared-expert gates only for the exact Qwen3.5 MTP class; pair weights/scales regardless of input order and reject incomplete pairs. Delegate to the original loader, retaining its mapper and reload behavior. Remove when affected checkpoints or upstream loading resolve the ignored-gate mismatch. Real MTP weight loading remains required. |
| `models/telechat3/telechat3_yarn.py`: `_get_and_verify_max_len` | **Retain.** Temporarily normalize TeleChat3's rotary name for target `config/model.py` validation, preserve nested `None` RoPE entries and restore caller-owned config afterward. An already-scaled maximum of 32768 with factor 4 remains 32768. Remove when upstream recognizes `telechat3-yarn` with these length semantics. |

## Removed and migrated patches

These entries are absent from the active patch tree and its initializer imports.
Migration destinations identify existing extension points, not additional active
monkey patches.

| Module / target | Decision, repair reason and removal condition |
| --- | --- |
| `models/minimax_m3/index_topk.py` | **Migrate to `vllm_metax/models/minimax_m3/common/ops/index_topk.py`.** Keep one-stage launches and the 16-row minimum decode tile. Inherit the complete v0.32 masked-pointer, score-buffer and split-K ABI changes; no upstream module/kernel replacement remains. |
| `models/minimax_m3/sparse_attn.py` | **Migrate to `vllm_metax/models/minimax_m3/common/ops/sparse_attn.py`.** Retain SUB_K=16 and minimum head tiles alongside target sentinel/causal/scale handling. Fix the all-empty merge locally; both the old patch and v0.32 produced NaNs for padded rows. |
| `models/minimax_m3/load_weights.py` | **Migrate to native backbone/MTP loaders and `metax/weight_loading.py`.** Keep destination-dependent inverse scales and interleaved hybrid coverage. Retire patch decorators, captured-original metadata and activation imports; top-level native text/VL methods own their loading sessions. |
| `models/minimax_m3/runtime.py` | **Migrate to `metax/model.py`.** Retain FP32 Gemma residual accumulation and standard TP reduction, but retire the global replacement of `fused_allreduce_gemma_rms_norm`; only MiniMax now uses the local helper. |
| `models/minimax_m3/dspark.py`, `models/minimax_m3/__init__.py` | **Migrate hidden-state capture into the native model; remove model-patch package and activation.** The deferred-FFN tap still needs a clone reduction. Only pre-model config hooks remain, under `bugfix/minimax_m3_draft_config.py` and the existing shared draft override. The old standalone DSpark path is a compatibility export. |
| `bugfix/qwen/qwen4_exp.py`: `kernel_warmup`; retired `_CONFIG_REGISTRY` override | **Migrate warmup; remove patch and activation import.** `models/qwen4_exp/metax/warmup.py` now implements an upstream `VllmJitKernel` provider. The MetaX QSA metadata builder records raw/aligned block-table widths; `Qwen4ExpQSAAttention.__init__` registers the layer while either runner activates `JitWarmupRegistry` during model loading. At warmup, the provider reads the now-bound cache and creates immutable shape/stride/dtype, query-head and selection-width keys. The default KV binding method is unchanged. The upstream warmup entry owns the JIT switch and `process_local_only` behavior; no model scan or global replacement remains. Equal layouts deduplicate through the registry. Compile-only meta tensors preserve real allocation strides without allocating another KV cache. The production fallback indexer retains on-demand JIT. Transformers continues to own config validation and normalization; the obsolete config override stays removed. Registration and real compile checks are recorded in [the migration report](../../.metax-upgrade/patch-review-20261103/QSA_REGISTRY.md); they do not establish checkpoint, graph or model numerical correctness. |
| `bugfix/minimax_m3/video_processor_preprocess.py`: `MiniMaxM3VLVideoProcessor._preprocess` | **Remove.** Target commit `b558f160a2c0abcb5902acc3c91a14c38a4af173` removes the required `do_convert_rgb` argument in `transformers_utils/processors/minimax_m3.py`. Transformers owns RGB conversion before this call. The shim's removal condition is met; its activation import is gone and callers use the upstream class method directly. The isolated retirement probe does not establish full video-model correctness. |
| `enhancement/engram_config.py`: `EngramConfig.verify_model_config` | **Remove.** Target `config/engram.py` no longer restricts model verification to CUDA/ROCm, so OOT acceptance needs no platform-check override. Preserve native model/parallel validation and shared-memory/THP configuration handling. The removal condition is met and the activation import is gone; this does not prove the separate Engram execution implementation. |
| `enhancement/quant_kernels/mxfp8_kernel.py`: OOT MXFP8 candidate registration | **Migrate.** Use target `register_linear_kernel(..., PlatformEnum.OOT, "mxfp8")` through `vllm_metax/registry/linear_kernels/mxfp8.py`, imported by the registry initializer after `_ensure_global_patch()`. The private-table patch/import is absent. OOT registration remains necessary; kernel correctness is a registry obligation. |
| `enhancement/deepseek_v41_support.py`: package export `DeepseekV4FP8Config` | **Remove redundant export patch.** Target `model_executor/layers/quantization/__init__.py` uses the V4.1 package branch only for CUDA/ROCm. MetaX registers its quantization implementation through the platform/quantization registry instead. The package assignment and activation import are absent; retain verification of the actual registered config in the registry's audit scope. |
| `enhancement/quant_kernels/moe_align_block_size.py`: prior run-only native ABI bridge | **Remove from the active inventory; current runtime disposition needs revalidation.** An earlier run introduced a seven-argument MCOPLIB/eight-argument upstream wrapper bridge. The current tree contains neither this module nor its activation import. Its historical component/model results cannot establish current ABI compatibility, and absence alone does not prove the underlying issue was fixed. Do not retain the old appended active-patch claim. |

## Shared infrastructure

| Module | Purpose and repair reason | Removal condition |
| --- | --- | --- |
| `utils.py` | Install module/class replacements, preserve descriptors and record captured originals. Explicit submodule mode redirects the import cache and parent package without executing the original leaf. Generic alias synchronization uses names starting with `vllm`, including MetaX modules; the separate submodule public-export scan uses `vllm.`. Promote an automatically synchronized alias to an explicit target once; reject repeated/conflicting explicit installation. Preserve the final Triton JIT/autotune object and shared class owner. | All runtime patches migrate to native extension points or another installer. |
| `__init__.py` | Expose the shared `patch` decorator without installing runtime patches itself. | No module imports this package-level export. |
| `bugfix/__init__.py`, `bugfix/int8_w8a8/__init__.py`, `bugfix/triton_support/__init__.py` | Activate shared correctness fixes, including global PP communication, fused-scale loading and rejection-sampler patches. | Their child patches have all been removed or migrated. |
| `models/__init__.py`, `models/deepseek_v4/__init__.py`, `models/joyai/__init__.py`, `models/kimi_k3/__init__.py`, `models/qwen/__init__.py`, `models/telechat3/__init__.py` | Activate model-owned patches, including the Kimi-K3 table guard. JoyAI uses the model key rather than the former `joyai_support` feature name. | Their child patches have all been removed or migrated. |
| `enhancement/__init__.py`, `enhancement/chores/__init__.py`, `enhancement/distributed/__init__.py`, `enhancement/MRV2/__init__.py`, `enhancement/quant_kernels/__init__.py` | Activate the current enhancement modules. Quantized-kernel activation now contains only the FP8-block patch; Engram, MXFP8 and the historical MoE ABI bridge are not imported here. | Their child patches have all been removed or migrated. |
| `enhancement/performance/__init__.py` | Import grouped routing, speculative batch ordering and conditional Gumbel sampling. | Their child patches have all been removed or migrated. |
| `template/class_method.py`, `template/module_attr.py` | Inactive descriptor-preserving/class and module-attribute examples, including alias synchronization and Triton decorator order. | The runtime patch mechanism is no longer needed. |

Generic alias synchronization uses the string prefix `vllm`, including
`vllm_metax` and other matching module names. Same-name MetaX bindings to the
original are therefore eligible; renamed references such as `_original_target`
and unrelated objects with the same name are preserved. The separate public-export
scan during whole-submodule replacement uses `vllm.` only. References inside local
variables or containers are outside both scans. Class-attribute replacement affects
the shared owner; it does not scan previously bound methods/property getters.
Installation, marker metadata and scan boundaries are documented in [README.md](README.md).

## Validation status

- The 2026-11-04 layout review covers staged changes and the final working-tree
  activation chain. Its [environment and check records](../../.metax-upgrade/patch-review-20261104/TODO.md)
  distinguish index evidence from final repairs. The index is preserved; repairs
  remain unstaged. No full-model numerical acceptance is implied.
- The inventory also accounts for three previously omitted existing modules:
  the MoE-LoRA shared-memory helper, MRV2 async-output option and INT8 indexer
  config extension. Their inclusion corrects documentation, not runtime support.
- Run artifacts under `.metax-upgrade/setup-preview/patch/` include the target mapping,
  kernel/retirement probes and per-commit history supplements. Their coverage and
  pending paths must be read explicitly; archived commits are not all completed reviews.
- Kernel launch recorders, AST comparisons and CPU stubs establish only their stated
  contracts. Native numerical checks remain required for the active kernel patches;
  distributed, PP/TP, speculative and graph configurations need their own evidence.
- Source, component, environment or contract changes invalidate affected prior checks.
  Neither the historical 0.30.0 audit nor a normal-entry run of an unrelated model
  certifies every patch in this inventory for v0.32.0.

The MiniMax native-model migration is reviewed against the same v0.32.0 NVIDIA
source, with its own [confirmed environment and current evidence](../../.metax-upgrade/minimax-m3-model-migration-20261104/TODO.md).
The [model adaptation record](../models/minimax_m3/README.md) details non-patch
contracts and validation limits; those model/kernel findings are outside this patch inventory.
