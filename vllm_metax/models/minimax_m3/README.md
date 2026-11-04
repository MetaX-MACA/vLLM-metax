# MiniMax-M3 on MetaX

The source contract is **vLLM v0.32.0**, commit
`5df63a7203cdb9328d824a27b682e77c7b9928be`. `common/` is copied from
`vllm/models/minimax_m3/common/`; `metax/` starts from the complete **NVIDIA**
model and MTP implementation. Unsupported NVIDIA CuTeDSL/CUTLASS/MSA kernel modules
are omitted and replaced by local Triton implementations. The AMD model, AITER kernels and shuffled cache layout are not
the adaptation baseline. Model-private imports resolve inside this package.

`ModelRegistry` registers the text model, VL model, MTP and
`DSparkMiniMaxDraftModel` here. The previous `minimax_m3_dspark.py` path only
re-exports the implementation. There is no MiniMax model monkey-patch package.

## Runtime contracts

| Area | NVIDIA contract and MetaX adaptation |
| --- | --- |
| Projections | Keep the fused sparse `q/k/v/index_q/index_k` projection, per-head Gemma QK normalization, partial NeoX RoPE and checkpoint shard names. |
| Main KV cache | Keep the NVIDIA logical `[block, kv_head, 128, 2 * head_dim]` layout and respect allocator strides. BF16 and dequantized FP8 E4M3/E5M2 sparse reads are exercised. |
| Index cache | Default to BF16, with a shared index-key vector per token and token-major shared top-k output. The native FP8-indexer capability guard remains in place. |
| Sparse kernels | Use the complete target kernels with 16-token K/V sub-tiles, a minimum 16-row dot tile and one-stage score launches for MACA resource limits. Preserve causal bounds, sentinel masking, scales and split-K score interfaces. |
| Normalization / TP | Accumulate Gemma residual sums in FP32. Keep each attention/FFN reduction with its designated following norm; use standard MetaX TP collectives. Auxiliary target taps reduce a clone so the live stream is not reduced twice. |
| Weights | Preserve `weight_scale_inv` only when direct/fused/expert destinations expose it; MXFP8 destinations still receive `weight_scale`. Apply the same destination rule to MTP. |
| Mixed quantization | Reuse the existing MXFP8 linear and packed NVFP4 expert registrations. Keep independent gate/up scales, local expert/shard coverage, interleaved text/VL loading sessions and explicit warnings for legacy input-scale defaults. |
| Native preprocessing | `metax/attention_ops.py` preserves the v0.32 Python signature and inspects the actual native schema. Installed MCOPLIB 0.4.14 has 23 arguments; source kernels have 25, including NVFP4 KV scales. Reject unknown schemas and unsupported scale arguments. |
| Vision | Keep the complete target vision tower, image/video preprocessing, projector and encoder graph interfaces. Generic platform-selected MetaX vision attention remains in use. |
| MTP / DSpark | Keep target MTP loading/PP interfaces. The existing MiniMax DSpark backport uses the v0.32 DFlash/DSpark lifecycle, Gemma norms, dense SwiGLU-OAI FFNs and standalone quantization exclusions. |

Draft configuration has an explicit MiniMax DSpark branch in
`patch/bugfix/minimax_m3_draft_config.py`. It resolves the bundled `dspark/`
checkpoint or accepts an explicit draft path, and uses that draft's quantization.
It reuses upstream TP, maximum-length and parallel-config helpers and keeps the
applicable DSpark checks and Pydantic validators. Other models and speculative
methods call the original initializer. This replaces the AST-based fallback
exemption; fixing configuration after the original initializer returns is too
late for its architecture inspection and parallel-config construction.

NVFP4 **expert weights** and NVFP4 **KV storage** are separate contracts. The
SM100 MSA/CUTLASS/CuTe modules are deliberately absent. Both prefill and decode
select portable Triton directly, even if a capability alias reports SM100.
Explicit `CUTLASS_MSA` and `TRITON_MSA` aliases resolve to the same local backend;
an explicit CUTLASS request logs the Triton substitution at construction time.
The indexer supports BF16; NVFP4 KV and FP8 indexer caches are rejected explicitly.
NVFP4 expert weights remain a separate existing quantization path. LoRA is not
advertised by the target model.

| Skipped NVIDIA implementation | MACA replacement |
| --- | --- |
| CuTeDSL index decode score | `common/ops/index_topk.py:minimax_m3_index_decode_score` |
| MSA prefill score and fused top-k | Local Triton `minimax_m3_index_score` / `minimax_m3_index_topk` |
| MSA sparse prefill attention | `common/ops/sparse_attn.py:minimax_m3_sparse_attn` |
| CUTLASS sparse decode | `common/ops/sparse_attn.py:minimax_m3_sparse_attn_decode` |
| MSA metadata and FP8 query buffer | Standard local Triton metadata and model-dtype queries |

## Patch review and dispositions

| Previous patch | Disposition and evidence |
| --- | --- |
| `index_topk.py` | Migrate launch adjustments into local kernels. Stage and minimum-tile restrictions remain necessary on MACA. Inherit v0.32's masked-pointer loads and updated split-K ABI instead of copying old kernel bodies. |
| `sparse_attn.py` | Migrate sub-tiles and head padding. Retain v0.32's safe decode sentinel handling (`1664c4f444`). The old patch and target merge both produced NaNs for all-empty padding rows; the local merge now returns zero, with a numerical regression. |
| `load_weights.py` | Migrate destination-dependent scale naming and strict hybrid sessions. Decorators, captured-original lookup and upstream alias replacement are obsolete once the model owns its methods. |
| `runtime.py` | Keep Gemma accumulation and TP semantics locally. The global replacement of the shared all-reduce/Gemma helper is unnecessary and removed. Other models keep their original helper. |
| `dspark.py` target tap | The inherited v0.32 auxiliary-state method still captures an unreduced deferred FFN result. Keep the clone reduction on the local model, alongside target PP relay behavior. |
| `dspark.py` configuration | Still required before model construction: v0.32 otherwise chooses the target checkpoint and rewrites an unknown DSpark architecture to DeepSeek. Retain only this bridge in `patch/bugfix/minimax_m3_draft_config.py`; the shared draft override preserves standalone architecture/quantization. Remove when upstream supports these configuration paths. |
| DSpark context projection | The old explanation that generic DFlash cannot handle quantized projections is outdated: v0.32 already dispatches through their quant methods. MiniMax's local context path remains necessary because its hidden/key norms use **`1 + weight`**, whereas Qwen's grouped context normalization uses ordinary RMS weights. |
| DSpark constructors | Refresh the old 0.30 constructor copies from v0.32, retaining MiniMax model selection and exclusion mapping. Preserve current draft-vocabulary rules and Markov gather/confidence-head setup. |
| Retired video-processor shim | Keep removed: target Transformers preprocessing already owns RGB conversion; the removed `do_convert_rgb` requirement must not be restored. |

All model/kernel behavior is implemented under `models/`. The remaining global
configuration hooks run before model selection and cannot be moved into a model
constructor. [Patch audit](../../patch/AUDIT.md) records their activation and
removal conditions separately.

## Validation and limits

The confirmed environment is MACA SDK/runtime 3.9.1.6, driver 3.8.23, two MXC500
GPUs (Torch reports `MetaX C500`), Python 3.12.11, MetaX Torch 2.10.0/Triton 3.6.0
and MCOPLIB `0.4.14+g5780a78.maca3.9.0.31.torch2.10`.

The run records native-kernel float64-reference comparisons, BF16/FP8 sparse
prefill/decode and graph replay, loading/configuration boundaries, normal-entry
two-layer dense+sparse-MoE BF16 dummy-weight generation, and synthetic MTP,
DSpark and VL construction. DSpark construction exercises local context-KV
projection and Gemma key normalization. TP reduction checks have a separate
two-rank check using standard MCCL with custom all-reduce disabled; the norm
and auxiliary tap matched the reference exactly.

No exact official MiniMax checkpoint/revision was supplied or present locally.
Synthetic configurations do not establish official layer counts, complete
real-weight loading, quantized-checkpoint accuracy, image/video generation,
full-model graph execution, PP/EP or MTP/DSpark serving. Those remain explicit
validation obligations rather than inferred support claims.

Reproduce the focused regressions with:

```bash
/opt/venv/bin/python -m pytest --confcutdir=tests/models tests/models/test_minimax_m3_metax.py -q
```

Durable [run state and evidence](../../../.metax-upgrade/minimax-m3-model-migration-20261104/TODO.md)
include the approved fingerprint, initial dirty-worktree archive, file mappings,
target-versus-local diffs, upstream commit records, frozen references/tolerances,
failed checks and current results. GPU checks require access to the MACA driver.
