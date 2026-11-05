# Attention Acceptance Cases

Select the cases affected by attention dispatch, layouts, indices, reductions or
metadata lifetime.

## Mathematical and routing checks

- For ordinary attention, use the correct causal/window/sink mask and compare both
  output and LSE. For indexers, compute weighted ReLU scores before top-k; compare
  selected sets with score ties handled rather than requiring arbitrary tie ordering.
- Exercise metadata builder -> wrapper -> kernel selection. Check that unsupported
  dtype/dimension/layout combinations are rejected before kernel launch.
- For graph changes, warm up, capture, change Q/cache/length/index buffer contents and
  replay. For batch invariance, execute the same request in different batch compositions.
- DCP claims require the relevant collectives, head gathering and rank-local causal/top-k
  behavior; a local merge reference does not exercise distributed communication.

## Discriminating cases

| Changed behavior | Useful cases |
| --- | --- |
| Batch split/reorder | Pure prefill, pure decode, mixed, short extends, uniform/variable MTP, zero-length padded requests and dense-MHA exclusion from MQA. Verify request/state alignment rather than token count alone. |
| Index conversion | Shuffled pages, crossing page boundaries, masked tails/interior holes and multiple prefill chunks/workspace starts. Preserve request-relative indices for consumers needing them. |
| Cache layout | Actual allocation/bound shapes, dtype, strides and storage offset; contiguous/interleaved layer pages. Poison unused gaps to expose physical-addressing errors and check aliasing for required zero-copy views. |
| BF16 indexer | Flat/paged separately, FP32/BF16 weights, masked intervals, multiple heads, tail tiles, empty ranges, 1D/per-query 2D context lengths and cross-page MTP. Repeat calls for numerical instability. |
| Quantized indexer | Actual quantization/insertion, page value/scale representation, weights with folded Q scales, flattened metadata and supported native next_n values. Mocked gather cannot exercise dequantization. |
| FA interfaces | Flat varlen, paged varlen and kvcache separately; equal/different QK/V dimensions; supported sinks and rejected dtype/dimension combinations; return-LSE and supplied-output contracts. |
| MLA LSE/DCP | Padded/real and gathered heads, log base, token/head axis order, empty-rank identity and rank-local causal bounds. |
| Graph/capture | Persistent metadata addresses, lazy scheduler initialization, changing contents with fixed shapes, padding and bounds changes after capture. |

For an installed attention-kernel defect, distinguish raw component behavior from
wrapper mappings using the same input shapes/dtypes, seeds and masks. Compare error
magnitude and repeatability; revalidate the exact fixed component build before removing
its numerical workaround.
