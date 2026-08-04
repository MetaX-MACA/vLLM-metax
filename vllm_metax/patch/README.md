# Runtime patches

This directory contains narrowly scoped runtime patches for behavior that cannot
currently be implemented through a supported vLLM or vLLM-MetaX extension point.
Typical uses are:

- working around an upstream vLLM bug;
- providing temporary MetaX compatibility for a new vLLM feature; or
- replacing a hot path while an equivalent upstream optimization is unavailable.

Monkey patches depend on private implementation details and can silently become
incorrect after an upgrade. Treat every patch as temporary, keep its scope small,
and give it an explicit removal condition.

## Prefer an extension point

Before adding a patch, check whether the change can use one of the supported
mechanisms instead:

- custom operators and pluggable layers: `vllm_metax/registry/`;
- attention backends: `vllm_metax/v1/attention/backends/` and
  `MacaPlatform.register_attention_backends()`;
- configuration or CLI adjustment:
  `MacaPlatform.check_and_update_config()` and
  `MacaPlatform.pre_register_and_update()`.

If none of these fits, explain why in the patch header.

## Directory layout

- `bugfix/` contains temporary fixes for upstream or integration bugs.
- `enhancement/` contains compatibility features that cannot use a registry or
  another plugin interface yet.
- `performance/` contains temporary performance replacements.
- `torch_fix/` is optional and reserved for `torch+metax` fixes. A torch fix must
  be imported explicitly from `vllm_metax/__init__.py` before other patches.
- `template/` contains one template for class methods and one for module
  attributes, including module functions, classes, and Triton kernels.
- `utils.py` provides the shared `@patch` decorator. A direct attribute name
  selects module handling, while a dotted class attribute path selects the
  independent class-method handling path.

Group related patches in a subpackage. Avoid adding unrelated replacements to a
single module.

## Adding a patch

1. Copy the matching file from `template/` into the appropriate directory
   and replace every `TODO`.
2. Record the reason, affected versions, upstream issue or pull request (when
   available), and a concrete removal condition.
3. Copy the complete target function to module scope, preserving its name,
   signature, unchanged structure, and return contract. This keeps later upstream
   diffs reviewable. Use `@patch("fully.qualified.target.module")` for module
   attributes and `@patch("module.path", "TargetClass.method")` for class
   attributes. This resolves the target class internally without importing it in
   the patch file.
4. Mark only the changed portion with the `MetaX Modification` comments shown in
   the template. The decorator retains the original callable and raises an error
   if the same target is patched more than once.
5. Import the module from the nearest `__init__.py`. Make sure its category is
   loaded by `vllm_metax.__init__._patch()`; importing a patch applies it
   process-wide.
6. Add a focused regression test that demonstrates both the original failure and
   the patched behavior. Run that test with every supported affected version when
   practical.

Do not catch broad import or attribute errors to make a stale patch appear to
work. A version mismatch should fail clearly during startup. If a patch is only
valid under a specific runtime condition, guard its registration with that
condition and test both branches.

## Review and removal checklist

A patch is ready for review when:

- the header is complete and the removal condition is actionable;
- the replacement preserves the target's public signature and return contract;
- duplicate patch attempts fail with a clear error;
- a regression test covers the changed behavior; and
- the implementation does not have a supported extension-point alternative.

Remove the patch, its registration import, and its regression-only compatibility
code as soon as the stated removal condition is met. Verify the upstream behavior
with the same regression test before deleting it.

## Supported targets

- Module functions are inferred from the replacement function name.
- Instance methods use an explicit path such as `"TargetClass.forward"`; the
  target class is resolved inside the patch utility.
- `staticmethod`, `classmethod`, and `property` descriptors are detected and
  preserved. The replacement may repeat the original decorator, with `@patch` as
  the outermost decorator, or omit it and let the class-handling path preserve the
  target descriptor.
- Module-level Triton kernels are replaced as callable objects. Put `@patch`
  outside `@triton.jit` and pass the target attribute explicitly when the JIT
  object does not expose the expected function name.

Targets must exist by default so misspelled paths fail during import. Use
`allow_missing=True` only for a compatibility patch that intentionally adds a new
module or class attribute.
