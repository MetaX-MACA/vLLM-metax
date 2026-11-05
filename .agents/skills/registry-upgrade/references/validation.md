# Registry Validation

## Registration and interfaces

- Fresh-process plugin startup using the normal entry point, followed by actual registry
  lookups. Record effective config/CustomOp/kernel classes and their source modules.
- Positive and negative dispatch cases: supported configuration, ignored layer, explicitly
  requested backend and unsupported combination. Ensure unrelated keys remain intact.
- Construct objects through the real registered constructor/factory, then exercise
  inherited consumers rather than stopping at signature binding.
- Exercise startup, lookup and final kernel selection as distinct registration stages;
  an earlier exception may have hidden a later failure.

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

For WNA16 numerical checks, dequantize packed weights with group scales and zero points,
then compute expert matmuls, activation and routed reduction. Include
nonzero zero points and multiple routed experts so a missing subtraction/scale is visible.

Test the actual JIT/precompiled branch selected by the environment. If both branches are
changed or advertised, validate both or state which remains unverified. Supporting the
Triton source does not establish MCOPLIB numerical correctness.
