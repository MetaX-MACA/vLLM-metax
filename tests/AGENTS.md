# Plugin test guidance

- The subject under test is `vllm_metax`, never unmodified upstream vLLM.
- Import plugin implementation directly from `vllm_metax.*` or verify plugin
  registrations through public extension points installed by the plugin.
- Upstream `vllm` imports are limited to interface types, public integration
  entry points and independent numerical or behavioral oracles.
- Do not keep a test whose only purpose is to validate upstream vLLM behavior.
- Tests for patched behavior must exercise the patch installed by `vllm_metax`
  and prove the observable behavior differs from, or preserves, the oracle.
- Source-kernel mode (`USE_PRECOMPILED_KERNEL=0`) is the default. Do not silently
  fall back to precompiled kernels or import `mcoplib`.
- Prefer real plugin objects and real tensor operations. Mock only external
  services or unavailable vendor boundaries with a documented skip condition.
- Test MACA-specific dispatch, schema, dtype, layout, stream, workspace and
  numerical contracts explicitly instead of inheriting CUDA assumptions.
