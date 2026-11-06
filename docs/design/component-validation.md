# Component experiments and acceptance scope

This proposal combines three bounded changes, not a completed compatibility
upgrade. The integration base is `0962c8735ffe4795800030680311410848512b82`.
Reference vLLM is `aa4022dc8176db8f769f93445d845f8755f9b4ca`; vLLM-Omni is
`4cbfcc3eb55597720b84339afab0842c03ac3c93`.

## Changes and safe defaults

| Component | Benefit | Boundary |
| --- | --- | --- |
| Attention | Avoid redundant LSE contiguous staging for stride-aware merge | MACA mcoplib keeps the old contiguous path; no default mcoplib speedup claimed |
| MoE dispatch | Lossless packing reduces small uneven gather launch count | Explicit experimental opt-in; four-rank measured envelope only |
| Sampling | Per-request generated-token window with eligible intervals, exemptions and three penalties | Explicit V1 processor load; standard penalties stay neutral; not V2 or a replacement for standard penalties |

Allocator lifecycle work is intentionally excluded. Its native failure/free
boundary needs a separate ABI review and real-device validation, including
existing native callback error branches. Its original commits and regression
tests remain in the separate allocator delivery.

## Historical component evidence

These measurements were collected on 2026-11-06 on Mars X203 with Torch
`2.6.0+mars3.5.3.9`, Triton `3.0.0+mars3.5.3.9`, HPCC 3.5.3 and HCCL. They are
**not** MACA/MCCL results. The target Torch `2.10.0+metax3.9.0.8`/MACA stack is
not installed on that host. No shared runtime packages were replaced.

### Attention

The standalone experiment covered 372 GPU boundary cases and four CPU provider
decision simulations. Two complete 180-row timing runs checked values and
changed-input graph replay. For the r4 samples where a copy was removed,
whole-boundary geometric-mean speedups were 1.360x (FlashAttention) and 1.245x
(FlashInfer); graph replay was 1.311x and 1.376x. Worst wall-time regression was
12.25% in r4 and 19.53% in r3; singleton cases had no reliable benefit.

The experiment used extracted upstream merge functions, not actual backend/DCP
or model execution. FlashInfer log2-to-ln conversion is unchanged. Native source
kernel ABI and real MACA backend acceptance are still required.

### MoE communication

Two- and four-rank HCCL suites each passed 252 cases, covering mixed dtypes,
integer extremes, uneven/zero token counts, strided inputs, repeated calls,
streams and changed-input graph replay. Failure/timeout cleanup was checked;
native teardown warnings were retained. This does not establish MCCL behavior.

Four-rank timing used 100 samples per trial, seven trials, maximum-rank wall
latency and both path orders. Pack/unpack and allocation are included:

| Sizes | Eager dispatch + combine, us | Reversed order, us |
| --- | --- | --- |
| `[0,0,0,8]` | 476.0 to 418.4 | 531.3 to 465.7 |
| `[0,1,2,3]` | 517.8 to 510.4 | 493.2 to 464.2 |

Whole graph replay improved from 148.7 to 88.9 us and 191.5 to 107.6 us.
However, reversed-order dispatch alone for `[0,0,0,8]` regressed from 350.5 to
388.0 us. Uniform and larger batches regressed, motivating the restricted gate.
For the measured four-tensor uneven case, broadcast calls fell from 16 to 4 in
dispatch (20 to 8 including combine). Payload did not shrink; maximum-rank
temporary storage increased from 67,072 to 200,192 bytes. The numeric-pack
comparison is a Torch proxy, not the installed mcoplib implementation.

### Sampling

Standalone exact-function checks passed 144 GPU cases. An independent Torch
2.14 CPU environment passed 121 tests with 20 GPU-only skips, including the real
V1 loader, request conversion and greedy sampler with the unmodified upstream
CPU sampling translation unit. That component build is not a full native vLLM
build, HTTP server or model generation test.

Twenty-eight benchmark configurations compared equivalent windowed policies,
not upstream full-history penalties. Each path has 90 measured samples. FP32
apply-only speedups were 2.86-17.60x; whole-step CPU-build/transfer/apply speedups
were 0.77-3.80x. Representative results:

| Dtype | Batch / vocabulary / window | Dense us | Sparse fused us | Speedup |
| --- | --- | --- | --- | --- |
| FP32 | 128 / 151936 / 32 | 12299.94 | 3236.85 | 3.80x |
| FP32 | 32 / 32000 / 128 | 2287.48 | 2953.54 | 0.77x |
| FP16 | 32 / 32000 / 128 | 2377.15 | 3105.55 | 0.77x |
| BF16 | 32 / 32000 / 128 | 2431.14 | 2981.30 | 0.82x |

Logical state uses 24 bytes per unique update. Batch128/vocabulary151936/window32
used 98,304 bytes versus 97,239,040 for dense counts plus mask. Conversely,
batch8/vocabulary256/window4096 used 49,152 versus 10,240 bytes (4.8x larger).
Host buffers, temporary allocations and allocator peak memory are excluded.
CPU rebuilding can dominate. No universal speedup or quality gain is claimed.

## Evidence preservation

The one-off launchers, source-extraction harnesses and machine inventories were
removed from this review diff, not erased from the original deliveries:

| Direction | Original delivery commit |
| --- | --- |
| Attention | `6861e8f7fe83956533607c6a99d109272a82d1fc` |
| Communication | `405639a1d7f389e6f47314e4d65ef8006cc7e004` |
| Sampling | `23a2a1788df95bcaddb677804bf61ca4b1d956bd` |

Raw samples, failed runs, source hashes and logs are retained outside the source
tree. The communication evidence archive SHA256 is
`ee30f4f25c97a691a0a538e4b0e6f9413f6bfe2936847af7fdd6c88730231194`.
Historical figures must not be presented as a fresh final-source MACA benchmark.

## Production acceptance still required

Integration review reran the portable sampling core and benchmark contracts on
the image's Torch 2.6 CPU path: **102 passed, 20 GPU-only skipped**. The final
communication functions passed **16 local tests** plus the uneven/all-zero
collective test on each of four HCCL ranks, using the newly retained regression
tests. The external HPCC harness replaces only the unavailable
`vllm.triton_utils` import with direct Torch/Triton dependencies and verifies
that all function ASTs are identical. This is still not production import or
manager integration coverage. All GPU runs used the shared owner launcher.

The integration run's full CPU adapter and attention suite could not complete:
the isolated CPU environment stalled reading the shared Lustre filesystem.
Earlier passes above are historical evidence, not a fresh integration pass.
Changed-file pre-commit hooks passed. The full `pre-commit run --all-files`
passed all checks except `pip-compile`, which failed resolving PyPI hostnames.
No hook was disabled and no dependency lock was changed to hide that failure.
The isolated test environment and dependency-lock check must be rerun.

- Run attention tests with both source and mcoplib providers on MACA, including
  actual DCP/FlashAttention/FlashInfer integration and model outputs.
- Validate MoE manager dispatch, MCCL uneven gather, graph lifetime, multi-node
  fallback and teardown; measure against the real installed baseline.
- Run the V1 sampling processor through actual serving and generation. Keep V2,
  speculative decoding and model-internal Omni codec sampling unsupported.
- Re-run whole-component timings on the final source with reserved hardware.
  Retain losses as well as gains; do not extrapolate to model throughput.

Until these checks pass, the PR is a draft for component review, not production
acceptance. Unit and standalone component passes do not waive these requirements.
