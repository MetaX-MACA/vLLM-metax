# Registry Validation

Generate independent, self-contained validation scripts as needed. Do not import the
repository's tests, fixtures or conftest. Use the intended interpreter and actual source
snapshot. Select cases based on the changed behavior, not an exhaustive fixed matrix.

## Registration and interfaces

- Fresh-process plugin startup using the normal entry point, followed by actual registry
  lookups. Record effective config/CustomOp/kernel classes and their source modules.
- Positive and negative dispatch cases: supported configuration, ignored layer, explicitly
  requested backend and unsupported combination. Ensure unrelated keys remain intact.
- Trace and check real constructor/factory signatures. Signature binding is an interface
  check only; construct real objects and exercise their inherited consumers when feasible.
- Reproduce all layers of previously reported failures. A successful direct module import
  is weaker than normal startup, and a successful lookup is weaker than kernel selection.

## Quantization state and layouts

- Allocate actual weights through the registered method and verify checkpoint orientation,
  packing factor, TP group boundaries, dtypes, strides and inherited state.
- Load controlled packed values. Compare conversion against independent bit extraction
  or dequantization, not a copy of the production reshape sequence. Include negative int32
  bit patterns, nonuniform scales/zero points and relevant singleton/multiple-group cases.
- Verify both w13 and w2, symmetric and relevant asymmetric branches, converted parameter
  identities, aliases and the quant config consumed by setup. A passing conversion helper
  does not prove the layer retained its outputs.
- For packed reinterpretation changes, cover contiguous and reachable strided inputs,
  single/multiple experts and single/multiple groups on CPU and available MACA GPUs.
- If production selection rejects a scheme, isolate post-load handling only with an
  explicitly recorded process-local selector/setup override. Keep conversion/scatter/config
  real. Report separately that the production scheme remains unsupported.

## Runtime behavior

For kernel or support-declaration changes, exercise the actual selected MetaX kernel
against an independent reference. For WNA16, dequantize packed weights with group scales
and zero points, then compute expert matmuls, activation and routed reduction. Include
nonzero zero points and multiple routed experts so a missing subtraction/scale is visible.

Test the actual JIT/precompiled branch selected by the environment. If both branches are
changed or advertised, validate both or state which remains unverified. Supporting the
Triton source does not establish MCOPLIB numerical correctness. EP/DCP, graph replay and
performance claims require their corresponding runtime checks; single-device tests do
not establish them.

When startup fails due to an unrelated optional dependency or ABI issue, capture the
original failure. A temporary process-local workaround can isolate the reviewed path;
record its exact scope without silently modifying installed packages.

## Completion evidence

Run appropriate syntax, formatting/lint and diff checks for changed files. Report tools
that are unavailable instead of implying they passed. Preserve commands and generated
script locations with outcomes and limitations. Separate source inspection, signature
checks, isolated layout/state tests, real kernel numerics and full-model validation.
