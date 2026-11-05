---
name: csrc-upgrade
description: Upgrade all MetaX native operator sources, headers, bindings and build contracts against the selected vLLM target. Review CUDA-to-MACA ports, ABI/schema changes, local and mcoplib dispatch, native build provenance and component numerics.
---

# MetaX native operator upgrade

Read [common-sense](../common-sense/SKILL.md) first. It owns upstream-first source
style, environment preparation, dependency closure, chip limits and durable acceptance.
Use this skill for `csrc/`, required `cmake/` files, native portions of `CMakeLists.txt`
and `setup.py`, and their Python operator schemas/wrappers. Coordinate shared files
with registry/attention/model/patch owners; do not silently take duplicate ownership.

## Complete native source coverage

**Upgrade every csrc file whether the run uses mcoplib or local source kernels.**
Include inactive kernels, C++/CUDA headers, shared utilities, bindings, generated-source
inputs, allocator and communication code. mcoplib selection never exempts local sources
from the full inventory, target mapping, source adaptation or verification obligations.
An unchanged file still needs a reviewed target-aligned retain decision.

Map moved/removed upstream native files and their new bindings/build targets. Follow
the exact confirmed upstream revision and corresponding CUDA implementation; "latest"
means that target, not an unrelated newer main branch. Review complete target modules
and the prior MetaX changes, then reapply necessary MACA differences under common rules.
MetaX-only kernels need an explicit platform-only mapping and maintained callers.

Use `--owner csrc-upgrade` in the shared run. Adapt native dependencies with the earliest
consuming platform/registry/attention phase, and complete the remaining native sweep in
framework before models. This skill does not introduce another phase or open early model
execution. Read-only model contract analysis informs native interfaces before model ports.

## Trace each operator contract

Record the chain from Python wrapper/CustomOp lookup through schema, registration,
binding and compiled target to the actual device implementation and its consumers.

- Review Torch stable-lib ABI versus ordinary Torch bindings, extension names and
  initialization, dispatcher namespaces/keys, schema defaults, optional arguments,
  mutation/alias annotations, return ownership and fake/meta registration.
- Check dtype and quantization format, shape/stride/contiguity, packed axes, alignment,
  block sizes, scale/zero-point conventions, index units/types and cache write/read
  agreement. Quantized KV/indexer kernels must match attention insertion, gather,
  prefill and decode consumers; model-specific differences belong in that model's spec.
- Check stream/device guards, asynchronous lifetime, scratch/workspace ownership,
  synchronization, graph capture and collective ordering. Review CUDA/MACA intrinsics,
  architecture guards, compiler flags and available installed library APIs.
- Update shared headers and indirect native dependencies with consumers. Review both
  operator availability and numerical semantics; registration success is insufficient.

## Respect the selected kernel provider

[environment-check](../environment-check/SKILL.md) must explicitly record whether this
run depends on mcoplib. Do not infer the choice from package presence.

| Provider | Required behavior |
| --- | --- |
| `mcoplib` | Build/run with `USE_PRECOMPILED_KERNEL=1`; use the selected environment's locked mcoplib, without replacing or upgrading it. Verify its actual schemas, callables and origins against new wrappers. Still upgrade and validate every local native source. |
| `source` | Build/run with `USE_PRECOMPILED_KERNEL=0`; rebuild local native extensions and prove that execution uses them. Do not fall back to mcoplib or report a precompiled binary as source validation. |

Source mode means no execution dependency on mcoplib, including its Python/Triton
helpers. Trace optional imports and patch paths as well as `_C`/MoE dispatch; a sampler
may import mcoplib independently of the main flag. Port necessary implementations or
use verified existing local/public-framework equivalents. Keep an already-installed
mcoplib untouched; its presence must not affect source-mode dispatch.

In mcoplib mode, compile/check upgraded local sources in an isolated build directory
and fresh component subprocesses. Record that source-validation context separately;
never overwrite the selected production installation or image mcoplib to test local
code. Include these isolated checks in the initial preparation/validation plan. They do
not change the selected provider for production or model checks.

## Native build and evidence

Inspect CMake source lists, includes, compile definitions, architecture selection, linked
libraries, stable ABI options and extension packaging. Match the Python import name to
the produced binary. Do not rely on an old binary or cached precompiled CMake setting:
retain the build command, effective `USE_PRECOMPILED_KERNEL`, relevant CMake cache,
compiler/SDK/Torch identity, source hashes and actual extension path/hash. Use a separate
build directory or targeted rebuild instead of deleting unreviewed user build outputs.

Compile success, symbol loading and schema review are distinct evidence. Use direct
component contract and independent numerical checks for changed operators, including
boundary sizes, invalid configurations, dtype/layout/scale/index cases and applicable
stream/graph/collective behavior. Test native callables from the recorded new build;
mcoplib output alone cannot validate the copied local implementation. Apply common's
current-chip rules to unavailable numerical branches and list them explicitly unverified.

For source mode, validate in fresh processes where mcoplib is unavailable or imports are
rejected, and inspect loaded modules/extension origins. A successful run with mcoplib
installed and unobserved is insufficient proof of independence. Registry/attention
wrappers must reach the tested source kernel through their normal supported dispatch.

Follow the existing successful-install prerequisite: an initial source-mode installation
that cannot compile blocks adaptation; this skill does not authorize compatibility edits
before that gate. Later source rebuilds use the same approved plan and invalidate affected
evidence. Complete-model smoke/e2e/batched validation remains in the final phase.
