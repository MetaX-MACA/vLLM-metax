# Runtime patches

Runtime patches provide temporary vLLM fixes or MetaX compatibility changes when
an existing registry or extension point cannot be used.

> Patches depend on vLLM implementation details. Keep them small, tested, and
> easy to remove after the upstream issue is fixed.

## Quick start

1. Choose a template:
   - `template/module_attr.py` for module functions, classes, or Triton kernels.
   - `template/class_method.py` for instance methods.
2. Copy the template into `bugfix/`, `enhancement/`, or `performance/`.
3. Complete the header and replace every `TODO`.
4. Import the new patch module from the nearest `__init__.py`.
5. Add a focused regression test.

Importing a patch module applies it immediately. Patching the same target twice
raises `RuntimeError`.

## Common examples

### Module function

The target name is inferred from the replacement function name:

```python
from vllm_metax.patch.utils import patch


@patch("vllm.some_module")
def target_function(arg: int) -> int:
    # Copy the complete upstream implementation and mark the MetaX change.
    ...
```

Specify the attribute when the names differ:

```python
@patch("vllm.some_module", "target_function")
def metax_target_function(arg: int) -> int:
    ...
```

### Class method

Use a dotted class attribute path. The target class is resolved internally and
does not need to be imported by the patch file:

```python
@patch("vllm.some_module", "TargetClass.forward")
def forward(self, hidden_states):
    ...
```

Keep `@patch` outside method descriptors:

```python
@patch("vllm.some_module", "TargetClass.from_config")
@classmethod
def from_config(cls, config):
    ...
```

The same form supports `staticmethod` and `property`.

### Triton kernel

`@patch` must be the outermost decorator so the final Triton object is installed:

```python
@patch("vllm.some_module", "target_kernel")
@triton.heuristics({...})
@triton.autotune(configs=[...], key=[...])
@triton.jit
def target_kernel(...):
    ...
```

## Patch layout

- `bugfix/`: upstream and integration fixes.
- `enhancement/`: temporary compatibility features.
- `performance/`: performance replacements.
- `template/`: ready-to-copy patch examples.
- `utils.py`: the shared `@patch` implementation.

## Keep in mind

- Prefer `vllm_metax/registry/`, attention backends, or platform configuration
  hooks when they can implement the change.
- Preserve the upstream name, signature, decorators, and unchanged code so future
  version updates are easy to diff.
- Mark only changed lines with the `MetaX Modification` comments from the
  templates.
- Use `allow_missing=True` only when intentionally adding a new compatibility
  attribute; normal patches require the target to exist.
- Record affected versions and a concrete removal condition in every patch.
