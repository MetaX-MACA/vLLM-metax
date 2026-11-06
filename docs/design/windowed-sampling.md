# Opt-in scoped windowed sampling penalties

This additive V1 logits processor applies a request-specific policy to the last
`window` generated tokens. It does not change the standard sampler or register
itself automatically. Token ranges are supplied by the caller: the processor
does not discover modalities or infer a tokenizer's audio/image token layout.

## Activation

In an installed compatible vLLM/MetaX environment, explicitly load
`vllm_metax.v1.sample.windowed_penalty_processor:WindowedPenaltyLogitsProcessor`
through `--logits-processors` or the `LLM(logits_processors=[...])` argument.
Select the V1 model runner with `VLLM_USE_V2_MODEL_RUNNER=0`. The fixed upstream
also has a **different V2 processor interface** under
`vllm/v1/worker/gpu/sample/logits_processor/`; this class does not implement it
and must not be presented as V2-compatible. Do not override a deployment that
requires V2 merely to enable this optional feature.
Set `SamplingParams.extra_args["metax_windowed_penalty"]` to this dictionary:

```python
policy = {
    "window": 128,
    "repetition": 1.2,
    "frequency": 0.1,
    "presence": 0.2,
    "token_ranges": [[100, 200], [300, 400]],
    "exempt_token_ids": [150],
}
```

For the OpenAI-compatible request schema, encode the policy as a **JSON string**
inside `vllm_xargs`, not a nested object. For example, the HTTP request field is:

```json
{
  "vllm_xargs": {
    "metax_windowed_penalty": "{\"window\":128,\"repetition\":1.2,\"token_ranges\":[[100,200]],\"exempt_token_ids\":[150]}"
  }
}
```

The ranges above are illustrative, not token IDs for a particular model. Supply
real ranges and control-token exemptions from the model's tokenizer. Leave the
standard repetition, frequency and presence penalties at `1`, `0`, `0`;
combining nonneutral standard penalties with this policy is rejected. Requests
without the key retain standard behavior even when the processor is loaded.

## Semantics and lifecycle

- Select the last `window` generated tokens first, then count eligible tokens.
  Prompt tokens are excluded. Tokens outside a range still consume window space.
- Ranges are half-open `[start, end)`, sorted on input, and cannot overlap.
  Omitting ranges or providing an empty list selects the whole vocabulary.
  Explicit exemptions override ranges.
- For each eligible token occurring `count` times, apply repetition once:
  `logit / repetition` when positive, `logit * repetition` otherwise, then
  subtract `frequency * count + presence`. Negative offsets are allowed.
- Validate window `[1, 65536]`, repetition `[1e-6, 1e6]`, frequency/presence
  `[-2, 2]`, at most 256 ranges and 4096 exemptions. Reject nonfinite values,
  booleans as numeric parameters, malformed JSON and unknown fields.
- Snapshot the policy but retain the engine's live output-token list reference.
  Apply removals, additions and moves in the documented V1 order. Reused slots,
  inactive replacement requests, swaps, rollback and same-length history
  replacement must not retain previous token counts.
- Build sparse state inside `apply()`, not `update_state()`: async scheduling
  fills output placeholders immediately before sampler execution, after batch
  metadata updates. This also avoids stale histories when membership is unchanged.

## Execution and limitations

One entry is stored for each unique `(row, token)` coordinate. Two int64 indices
and two FP32 parameters use 24 bytes per entry on the device, excluding allocation
overhead and temporary host storage. No batch-by-vocabulary counter is retained.
This can use **more** state memory than a dense representation for a nearly full
vocabulary. CPU counters and pinned buffers are rebuilt for every sample; there
is no claim of constant CPU overhead or allocation-free execution.

The CPU path uses sparse PyTorch indexing. CUDA-like devices use a lazy-loaded
Triton gather/penalty/scatter kernel, supporting FP32/FP16/BF16 and row/column
strides. Arithmetic uses FP32 with a final cast. Coordinates must be unique;
the normal state builder guarantees this. Out-of-vocabulary coordinates are
masked. Logits must not have overlapping storage.

The referenced upstream V1 loader rejects custom processors with speculative
decoding or pooling. This implementation does not bypass that restriction. It
does not hook model-internal Omni codec samplers, provide per-modality independent
windows, promise CUDA-graph capture, or establish any output-quality benefit.
Loading a custom processor can add engine-side history synchronization even for
requests that omit the policy; no claim of zero engine overhead is made.

## Fixed-revision comparison

Reviewed references (not an assertion about all future upstream releases):

| Source | Commit |
| --- | --- |
| MetaX base | `0962c8735ffe4795800030680311410848512b82` |
| vLLM | `aa4022dc8176db8f769f93445d845f8755f9b4ca` |
| vLLM-Omni | `4cbfcc3eb55597720b84339afab0842c03ac3c93` |

In vLLM, `vllm/sampling_params.py` and
`vllm/v1/sample/ops/penalties.py` implement standard global-history penalties.
The generic V1 extension API is already upstream in
`vllm/v1/sample/logits_processor/interface.py`; this work uses that API, not a
new engine extension mechanism. The async ordering was inspected in
`vllm/v1/worker/gpu_input_batch.py` (`refresh_metadata`,
`update_async_output_token_ids`) and `gpu_model_runner.py` (`_sample`).

Omni **already has model-specific window penalties**. Yue2 defines
`KEY_PENALTY_WINDOW`, phase-specific vocabulary and windowed sampling in
`vllm_omni/model_executor/models/yue2/yue2.py`. MiniCPM-o provides
`apply_window_penalty` in
`vllm_omni/model_executor/models/minicpmo_4_5/codec_state.py`. These are prior art,
not missing features. Other repetition controls such as DRY also exist; this
feature is not equivalent to DRY and does not claim to invent repetition control.

The narrow addition is a reusable, explicitly loaded **per-request combination**
of a generated-token window, eligible token intervals, exemptions and all three
penalty types, implemented using sparse unique-coordinate updates. No equivalent
combination was found in the reviewed standard sampling paths. This is a
functional difference, not a measured quality improvement or an unconditional
claim that no third-party processor can implement it.

See [the fixed-revision comparison](windowed-sampling-comparison.md) for exact
code locations and [component measurements](component-validation.md) for the
benchmark setup, results and measured regressions.

## Validation and benchmarking

Run the standalone numerical tests without the repository's unrelated model
fixtures:

```bash
python -m pytest -q --confcutdir=tests/metax_sampling tests/metax_sampling/test_windowed_penalties.py
```

The adapter suite requires an actual vLLM installation; it does not install mock
modules or monkey-patch upstream imports:

```bash
VLLM_TARGET_DEVICE=cpu VLLM_PLUGINS= VLLM_USE_V2_MODEL_RUNNER=0 python -m pytest -q --confcutdir=tests/metax_sampling tests/metax_sampling/test_windowed_processor.py
```

The benchmark checks numerical agreement, performs five warmups per path, resets
logits outside the measured interval, synchronizes before/after each sample and
randomizes path order between repeats. It saves raw timing samples, median/p95,
source hashes and device/runtime identity. It compares equivalent **windowed**
dense PyTorch, sparse PyTorch and fused Triton implementations, with separate
apply-only and CPU-build/transfer/apply measurements. It is not a comparison with
upstream full-history penalties or an end-to-end model-throughput measurement.

```bash
python -m benchmarks.benchmark_windowed_penalties --iterations 30 --repeats 3 --output benchmark-fp32.json
python -m benchmarks.benchmark_windowed_penalties --batch-sizes 1 8 --vocab-sizes 256 --windows 128 4096 --output benchmark-crossover.json
```
