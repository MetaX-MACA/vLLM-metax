# MetaX Component Contracts to Recheck

These observations came from a C500 review on 2026-09-20 using `/opt/venv`:
`deep_gemm 0.2.0+maca0.4.11.1`, `flash_attn 2.6.3+metax3.8.2.2torch2.10`,
Torch `2.10.0+metax3.8.2.2`, Triton `3.6.0+metax3.8.2.2`.
They are not universal facts about future packages or other GPUs. Reproduce relevant
constraints with the active binaries before retaining, adding, or removing restrictions.
Local wrappers and fallback implementations may have changed since this review.

## FlashAttention

Inspect `vllm_metax/v1/attention/backends/fa_utils.py`, the installed
`flash_attn.flash_attn_interface`, and actual callers in regular/DiffKV attention,
MLA prefill, TurboQuant and context-parallel helpers.

| Local observation | Required review action |
| --- | --- |
| FA2 and FA3 distributions were both installed, but these paths imported `flash_attn` FA2. | Identify the selected callable; package presence and a version-selector integer do not identify the executing kernel. |
| MetaX FA2 supported QK/V `192/128`. | Do not copy upstream FA3/FA4-only DiffKV gates without testing. Distinguish exact dimensions from rounded dimensions. |
| Varlen accepted `return_attn_probs`, `block_table`, `s_aux` and `softcap`; it did not accept `fa_version`, `return_softmax_lse`, `out`, `output_scale` or `num_splits`. | Check every keyword on each reachable path, including batch-invariant and context-chunk paths. Inspect actual tuple returns and LSE layout. |
| Kvcache accepted `num_splits`, whereas varlen did not. | Capability is per entry point. Do not transfer flags between varlen and decode APIs. |
| BF16 sinks passed for `64/64` and `192/128`; QK size 128 and FP32 sinks were rejected. | Validate query dtype, sink dtype, QK/V dimensions and shape together. Reflect restrictions in selection and runtime validation; preserve parameter loading semantics if converting sinks. |
| The local DiffKV backend accepted the `192/128` configuration only. | Explain that excluding 64 here is a backend constraint; ordinary `64/64` sinks belong to the regular backend. Rounding 160 to 192 does not establish sink support. |
| Native MetaX varlen could not write a supplied `out`. | Do not advertise `supports_out()` merely because V is unpadded; validate the chosen callable's output-buffer contract. |

For batch invariance, omitting an unsupported keyword fixes an API error; it does not
by itself establish numerical invariance across batch compositions. Validate the
required invariance when claiming that capability. Likewise, an FA4 helper copied into
a file does not make FA4 features available. Verify feature gates for quantized KV,
quantized queries/output, sinks, multimodal masks, dynamic causal flags and scheduler APIs.

## DeepGEMM and indexer logits

Inspect both `vllm.utils.deep_gemm` and `vllm_metax.utils.deep_gemm`, the registered
sparse indexer CustomOp, and installed `deep_gemm` attention wrappers and kernels.
The upstream wrapper may transform arguments before forwarding them to MetaX.

Both paged and non-paged APIs compute indexer scores, not normalized attention:

`score[q,k] = sum_h(weight[q,h] * max(dot(Q[q,h], K[k]), 0))`

ReLU belongs before the head reduction. FP8/INT8 Q scales may already be folded into
weights; verify this along the caller chain.

| Local observation | Required review action |
| --- | --- |
| BF16 non-paged used Q `[M,H,D]` and one BF16 K tensor `[N,D]`, not `(values, scales)`. | Do not trust stale FP8-shaped BF16 docstrings or tensor annotations. Check actual argument objects and numerical output. |
| FP8/INT8 paths used a Q tensor; non-paged K was `(values, scales)`. Upstream unified FP8/FP4 APIs used different conventions. | Do not blindly introduce a Q tuple, FP4 dispatch, or a varlen `indices` argument. Trace the selected symbol, quantizer and scale representation. |
| Raw scheduling metadata accepted `context_lens, block_kv, num_sms, blocks_per_split`; it had no `indices` keyword. | Check the upstream wrapper: `indices=None` was omitted before forwarding and was safe; forwarding `indices=None` directly to the raw function would not be. |
| MetaX metadata sizing used `get_num_blocks_paged_mqa_logits_metadata(num_sms)` (then 32 times num_sms). | Use the actual sizing helper; verify buffer size and any upstream SM/next_n scheduling transformations rather than copying NVIDIA assumptions. |
| Native MTP support was deliberately disabled in the local builder. | Test actual `next_n` lengths, 1D/2D context bounds, scheduler metadata and kernels before enabling it. A callable signature is insufficient. |
| Non-paged BF16 produced large numerical errors even in a process without vLLM patches. | Compare the raw package and adapter separately against a reference. Keep or replace the local numerical fallback based on fresh evidence; successful dispatch is not a correctness verdict. |
| BF16 paged logits passed on contiguous cache but asserted `kv_cache.is_contiguous()`. FP8/INT8 paged tests passed with gaps between pages. | Validate layout support per dtype. Do not infer BF16 stride support from FP8/INT8. Restrict layout selection or use a verified stride-aware implementation; avoid copying the whole cache per step. |

BF16 paged K was `[pages,page_size,1,D]` in BF16, with no scale tail. Quantized
paged K used a logical byte tensor `[pages,page_size,1,D+4]`, but a physical page
stored all value bytes followed by all FP32 scale bytes. The logical last dimension
did not imply token-interleaved scales. Verify insertion and consumption together,
especially when stride(0) spans a multi-layer block slab.

Check backend availability separately from support. Functions named `*_maca` may be
stubs while the default dispatch uses Triton/TileLang. Conversely, an optional
package's presence does not validate its native kernels on the active GPU. Also verify
`has_deep_gemm`, platform capability helpers and feature environment flags: MetaX is an
out-of-tree platform, not automatically a CUDA SM90/SM100 device.

## FlashMLA and sparse MLA

Inspect the local `ops/flashmla.py` wrapper and the actual installed `flash_mla`
interface independently of FlashAttention. Do not conflate their FP8 support.

- Dense FP8 wrapper symbols may exist but raise `NotImplementedError`; sparse FP8 can
  have a separate usable implementation. Capability declarations must reflect that.
- Verify whether `get_mla_metadata` returns an initialized schedule or a mutable object
  initialized on first call. Check reuse restrictions on shape, lengths and top-k state
  across layers, warmup and graph replay.
- BF16 sparse decode consumes paged slots (`page * block_size + offset`). BF16 sparse
  prefill can read a zero-copy flat cache view and needs physical row offsets based on
  the page stride. These mappings differ when other layers occupy gaps between pages.
- FP8 separate prefill typically gathers/dequantizes into a workspace. One converter
  can produce decode paged slots and prefill workspace offsets using per-token mapping
  metadata. Do not copy that conversion strategy to BF16 without tracing both consumers.
- Check output/LSE tuple unpacking, actual versus padded heads, log base and layout
  separately for dense decode, sparse decode and sparse prefill. Sparse prefill's
  documented log2 LSE must not silently enter an ln-based merge.
- Review mixed versus separate batch support under DCP, required communication backend,
  gathered head count and rank-local top-k filtering. Empty local top-k rows need neutral
  outputs/LSE. Do not infer DCP support from a single-rank numerical test.

## Other local components

Inspect MetaX custom ops/MCOPLIB, Triton, Torch/MACA, communication helpers and their
extension paths when the affected path uses them. Verify argument semantics, ABI,
return values and capture behavior instead of assuming the upstream `_C` operation
has the same implementation. If an optional dependency fails before the attention
path runs, report that separately; a process-local workaround must be disclosed and
must not mock the component being validated.
