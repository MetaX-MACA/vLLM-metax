# Quantization Contracts and Review Patterns

These are investigation patterns from registry adaptation, not permanent declarations
of supported MetaX features. Reinspect the current source, installed APIs and selected
runtime path before applying them.

## Configuration to implementation

Trace quantization-name lookup through `get_quant_method` for ordinary linear layers,
RoutedExperts/MoE, ignored layers and KV-cache methods as relevant. Check online versus
serialized checkpoints, ignored-layer match modes and fused module mappings. A registered
configuration may still return an unadapted upstream method for one branch.

Inspect the selected oracle, enum identities, experts class and prepare/finalize factory.
For MetaX Triton paths, follow invocation to the JIT or precompiled MCOPLIB implementation
under the actual feature flags. Merely accepting `has_zp` or bias arguments does not prove
that both implementations produce correct results. Keep unsupported combinations rejected
until the required path has evidence; distinguish an intentional limitation from a defect.

## Inheritance and weight lifecycle

For WNA16 and compressed-tensors adaptations, check the current parent constructor and
all inherited consumers before replacing initialization. Examples of relevant state are
`is_transposed`, `is_marlin`, `input_dtype`, group size, bit width, symmetry and activation
ordering. Their meanings must be derived from current callers:

- `is_transposed` can describe checkpoint allocation/loading orientation. A Triton
  runtime can require it to be true without using a Marlin kernel.
- Bypassing a parent's backend selection does not initialize its other state.
- A correct constructor signature is insufficient if `create_weights` then fails.
- A factory copied from upstream may accept different keywords from the local factory.

Follow this sequence with concrete tensor shapes and strides:

```text
allocate -> load/shard -> convert -> replace parameters/aliases
-> build quant config -> build kernel -> execute
```

Inspect symmetric and asymmetric branches separately. Conversion returning packed
zero points is ineffective if post-load code discards them and the quant config reads
old `layer.*_zero_point` parameters. Use the repository's parameter-replacement utility
where required for reload/alias behavior. Do not alias converted weights before replacing
the actual parameters, and do not assume checkpoint layouts equal kernel layouts.

Activation ordering aliases can be normalized by the installed compressed-tensors
package. Inspect the parsed value and required `g_idx` semantics before concluding that
removing a string check opens an unsupported path.

## Packed INT4 zero points and singleton strides

A common checkpoint layout is int32 `[E, G, N/8]`, with eight 4-bit zero points packed
per integer. A consuming kernel can require uint8 `[E, N/2, G]`, two zero points per byte.
Verify this against its pointer arithmetic, including scale and expert strides.

For ordinary contiguous checkpoint tensors, expanding bytes before transposition gives:

```python
# [E, G, N/8] int32 -> [E, G, N/2] uint8 -> [E, N/2, G]
converted = packed.contiguous().view(torch.uint8).transpose(1, 2).contiguous()
```

The old transpose-first sequence can fail when `G == 1`: `[1,1,32]` with stride
`[32,32,1]` becomes `[1,32,1]` with stride `[32,1,32]`. It is considered contiguous
because the last dimension is singleton, so `.contiguous()` need not copy. Dtype
reinterpretation still requires last stride 1. Do not assume contiguity proves all
stride requirements. Check storage offset/alignment and any other singleton packed
axis allowed by the actual shape constraints; the example is not a universal repair
for arbitrary tensor layouts.

Byte expansion must preserve bit order and signed int32 bit patterns. Validate using
independent shifts/masks, including negative packed integers and nonzero, nonuniform
zero points. All-zero tensors only verify shapes. In the expand-first formulation,
explicit E/G/packed-N variables are unnecessary because view and transpose preserve
and transform those dimensions directly. Derive INT8 packing separately if supported.

## FP8 and INT8 MoE

- Check helper imports against the actual module, including dynamic exports. An unused
  import of a nonexistent helper can prevent the whole registry from loading.
- Verify FP8 checkpoint block shapes versus TP shard dimensions, any refined block
  shapes, scale upsampling, encoded quant keys and the selected kernel's supported
  shapes. Do not declare arbitrary blocks supported because one upstream kernel does.
- Static/dynamic and per-tensor/per-token activation scales are distinct contracts.
  Trace scale ownership and `None` semantics through quant-config creation and kernels.
- For added INT8 bias arguments, inspect the local helper, stored quant config and
  selected experts implementation, then validate nonzero bias numerically.
- DeepGEMM and other MetaX component APIs can differ from the upstream wrappers. Record
  actual callable and extension origins and inspect argument semantics rather than
  inferring support from a package name or availability check.
