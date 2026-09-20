# Validation That Distinguishes Real Failures

Select tests for the affected behavior. Do not require every model/configuration for
an unrelated small edit, but do not replace numerical validation with import checks
when changing kernel dispatch, tensor layouts, indices or reductions.

## Layer the evidence

1. Import in a fresh process; verify actual source/binary origins and registration.
2. Check callable signatures and execute valid/invalid configuration probes. For
   selection changes, verify rejection occurs before an unsupported kernel is launched.
3. Compare real kernels and wrappers with an independent reference. For ordinary
   attention, apply the correct causal/window/sink mask and compare output and LSE.
   For indexers, compute weighted ReLU scores before top-k; test the selected set as
   appropriate, allowing score ties rather than demanding arbitrary order.
4. Exercise metadata builder -> wrapper -> kernel integration. A kernel test bypassing
   the constructor does not prove constructor or backend capability declarations correct.
5. When affected, warm up, capture, change Q/cache/length/index buffers, replay and compare
   again. For batch invariance, compare the same request in different batch compositions.
6. Run suitable model/distributed tests for claims that depend on those layers. Clearly
   state when a full model, real quantized storage path, distributed collective or
   performance benchmark has not been tested.

## Discriminating cases

| Changed behavior | Useful cases |
| --- | --- |
| Batch split/reorder | Pure prefill, pure decode, mixed, short extends, uniform and variable MTP, zero-length padded requests, dense-MHA exclusion from MQA. Verify request/state alignment, not just token count. |
| Index conversion | Shuffled pages, crossing a page boundary, masked tails and interior holes, multiple prefill chunks/workspace starts. Preserve original request-relative indices for consumers that need them. |
| Cache layout | Actual allocated shape, bound shape, dtype, strides and storage offset; contiguous and interleaved layer pages. Poison unused gaps to expose wrong physical addressing and check aliasing if a zero-copy view is required. |
| BF16 indexer | Flat and paged separately; FP32/BF16 weights; masked intervals; multiple heads; tail tiles; empty ranges; 1D and per-query 2D context lengths; cross-page MTP. Repeat calls to detect unstable numerical defects. |
| Quantized indexer | Actual quantization/insertion, page value/scale representation, weights with Q scale folded in, flattened metadata and supported native next_n values. Mocked gather cannot validate dequantization. |
| FA interfaces | Flat varlen, paged varlen and kvcache independently; ordinary and differing QK/V dimensions; supported sink combinations plus rejected dtype/dimension combinations; return-LSE and supplied-output contracts. |
| MLA LSE/DCP | Padded versus real heads, gathered heads, correct log base and token/head axis order, empty-rank identity, rank-local causal bounds. A local reference of the merge does not validate collectives. |
| Graph/capture | Persistent metadata addresses, lazy scheduler initialization, dynamic contents with fixed shapes, padding and bounds changes after capture. |

## Generate independent validation

Do not depend on the repository's `tests/` directory for now. Do not import its test
modules, reference helpers, fixtures or conftest files. Generate only the scripts or
tests needed to expose the affected behavior, using current production code and the
installed component interfaces as inputs to the investigation.

Create a fresh temporary directory for each task. Supply synthetic inputs, explicit
initialization and an independent numerical reference inside that directory. Prefer a
standalone Python script when pytest adds no value. If using pytest, make the generated
tests self-contained and isolate them from repository fixtures and unrelated plugins.
For example, after creating the indicated files in the task's temporary directory:

```bash
/opt/venv/bin/python /tmp/<task-directory>/validate_attention.py
# Alternatively, for generated pytest cases:
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /opt/venv/bin/python -m pytest \
  --noconftest -c /dev/null /tmp/<task-directory>/test_attention_contract.py -q
```

The paths above are placeholders, not existing artifacts. State the actual generated
paths and commands in the result, including any process-local setup or workaround.
Explicitly select and verify the edited or staged source in the generated process;
launching a script from /tmp must not silently select an old installed MetaX wheel.
Do not silently suppress a dependency failure from the attention component under test.
Do not add generated tests to the repository or stage them unless requested. Preserve
the reproduction artifact for the current handoff, but regenerate it in future sessions
rather than assuming an old temporary file still exists.

For a suspected installed-kernel defect, retain a minimal script importing only Torch
and that component. Record input shapes/dtypes, seeds, masking, error magnitude and
repeatability. Compare against an independent implementation, not a second wrapper
that calls the same faulty kernel. If the installed component is fixed later, verify
that exact build before removing the workaround. Record performance separately from
correctness; a validated fallback is not automatically production-performance equivalent.
