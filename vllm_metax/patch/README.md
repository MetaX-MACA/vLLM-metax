# Runtime patches

Runtime patches provide temporary vLLM fixes or MetaX compatibility changes when
an existing registry or extension point cannot be used.

> Patches depend on vLLM implementation details. Keep them small, tested, and
> easy to remove after the upstream issue is fixed.

The current upstream compatibility review and removal decisions are recorded in
[AUDIT.md](AUDIT.md).

## Quick start

1. Choose a template:
   - `template/module_attr.py` for module functions, classes, or Triton kernels.
   - `template/class_method.py` for instance methods.
2. Choose a directory using the classification rules below. Model-specific
   patches go in `models/<model>/`; shared performance patches go in
   `enhancement/performance/`.
3. Complete the header and replace every `TODO`.
4. Import the new patch module from the nearest `__init__.py`, then ensure
   that package is imported by its parent initializer.
5. Add a focused regression test.

Importing a patch module applies it immediately. Module-attribute patches also
update same-name exports in already-loaded modules whose names start with `vllm`
when they reference the original object. This includes `vllm`, `vllm.*`,
`vllm_metax`, and `vllm_metax.*`. Future imports resolve the replacement normally.
Unrelated objects with the same name are preserved. Renamed exports
(`from ... import target as other_name`) are not scanned; a same-name reference
inside a MetaX module is eligible for replacement too.

Explicitly patching the same target twice raises `RuntimeError`. Stacked
decorators may still name an automatically updated alias: the first explicit
installation preserves its original metadata, and subsequent explicit
installations fail. Class-method patches preserve descriptors on the shared
class; they do not scan module exports for bound methods or property getters.

## Installation flow

The decorator applies the replacement when the patch module is imported.
`replace_module=True` selects submodule replacement. Otherwise, a name such as
`target_function` selects a module attribute; `TargetClass.forward` selects a
class attribute. This diagram is also kept in the `utils.py` module docstring:

```text
                +------------------------------------------------+
                |   patch(module, attribute, ...)(replacement)   |
                +------------------------------------------------+
                                         |
                                         v
                            +------------------------+
                            |  replace_module=True?  |
                            +------------------------+
                                         |
             +---------------------------+-------------+
             | yes                                     | no
             |                                         v
             |                            +------------------------+
             |                            |  attribute has a dot?  |
             |                            +------------------------+
             |                                         |
             |                           +-------------+-------------+
             |                           | no                        | yes
             v                           v                           v
+------------------------+  +------------------------+  +------------------------+
| SUBMODULE REPLACEMENT  |  |    MODULE ATTRIBUTE    |  |    CLASS ATTRIBUTE     |
+------------------------+  +------------------------+  +------------------------+
             |                           |                           |
             v                           v                           v
+------------------------+  +------------------------+  +------------------------+
| Validate module inputs |  |  Resolve module/name   |  |  Resolve owner class   |
| Import parent package  |  |  Infer name if absent  |  |   and attribute name   |
+------------------------+  +------------------------+  +------------------------+
             |                           |                           |
             v                           v                           v
+------------------------+  +------------------------+  +------------------------+
|  Check loaded module   |  | Read original (static) |  | Read static descriptor |
| and parent duplicates  |  |                        |  |                        |
+------------------------+  +------------------------+  +------------------------+
             |                           |                           |
             v                           v                           v
+------------------------+  +------------------------+  +------------------------+
| If unloaded/required:  |  | Check target state [A] |  | Reject duplicate patch |
|  find_spec; skip leaf  |  | (may return or raise)  |  |                        |
+------------------------+  +------------------------+  +------------------------+
             |                           |                           |
             v                           v                           v
+------------------------+  +------------------------+  +------------------------+
|    Record original     |  |    Record original     |  |   Rebuild descriptor   |
|    Set sys.modules     |  |   Set module export    |  | (keep setter/deleter)  |
|  Set parent attribute  |  |                        |  |                        |
+------------------------+  +------------------------+  +------------------------+
             |                           |                           |
             v                           v                           v
+------------------------+  +------------------------+  +------------------------+
| If previously loaded:  |  |  If original existed:  |  |    Record original     |
|  Sync module/callable  |  | Sync same-name aliases |  |  Set class attribute   |
|  aliases by identity   |  |   by object identity   |  |                        |
+------------------------+  +------------------------+  +------------------------+
             |                           |                           |
             v                           v                           v
+------------------------+  +------------------------+  +------------------------+
|  Return module as-is   |  |  Return export as-is   |  |    Return installed    |
|                        |  |   Keep JIT/autotune    |  |    class attribute     |
+------------------------+  +------------------------+  +------------------------+
```

Missing attributes raise `AttributeError` unless `allow_missing=True`. Submodule
replacement checks the unloaded target's spec and raises `ModuleNotFoundError`
when absent, without executing the original leaf. On the class path, duplicate
detection checks the underlying implementation, and installation preserves the
existing descriptor and any property setter/deleter.

The module path has one exception to duplicate rejection: a stacked decorator
may explicitly name an alias that this installation already updated
automatically. That alias is promoted once, retaining its recorded original:

```text
Module target state [A]:

                  +------------------------------------------+
                  |  Same replacement AND automatic alias?   |
                  +------------------------------------------+
                                       |
                 +---------------------+---------------------+
                 | yes                                       | no
                 v                                           v
+----------------------------------+        +----------------------------------+
|    Promote alias to explicit     |        |      Target is replacement       |
|      Keep recorded original      |        |       OR has patch marker?       |
|     Remove automatic marker      |        |                                  |
|        Return immediately        |        |                                  |
+----------------------------------+        +----------------------------------+
                                                             |
                                                   +---------+---------+
                                                   | yes               | no
                                                   v                   v
                                            +--------------+    +--------------+
                                            | RuntimeError |    |   Continue   |
                                            |              |    | installation |
                                            +--------------+    +--------------+
```

For example, if two modules export the same original object `O`, patching its
defining export to `R` updates both bindings and records both originals:

```text
+------------------------------+    +------------------------------------------+
|            Before            |    |     After patch(source, "target")(R)     |
|                              |    |                                          |
|      source.target -> O      |    |            source.target -> R            |
|   vllm.caller.target -> O    |    |         vllm.caller.target -> R          |
| MetaX._original_target -> O  |    |       MetaX._original_target -> O        |
+------------------------------+    +------------------------------------------+

                                    +------------------------------------------+
                                    |          R.__vllm_metax_patch__          |
                                    |                                          |
                                    |            source.target -> O            |
                                    |         vllm.caller.target -> O          |
                                    +------------------------------------------+
```

The generic alias scan visits existing entries in `sys.modules` whose names
satisfy `module_name.startswith("vllm")`. This is a string-prefix check, so it
includes both `vllm` and `vllm_metax`, as well as names such as `vllm_extra`;
it is not restricted to the exact two package namespaces. It does not import
additional modules. A binding is updated only when its attribute name matches
the patched name and its value **is** the original object. Alias updates record
metadata directly without recursively calling `patch()`.

Future imports obtain `R` from the defining export. Renamed exports and
references inside local variables or containers are outside this scan. Thus
`vllm_metax.caller.target` can change from `O` to `R`, while a separately named
`vllm_metax.caller._original_target` remains `O`. Class aliases share their owner
class, so patching a class attribute updates that owner without scanning module
exports.

There are two scan boundaries in the current implementation:

| Scan | Loaded module-name filter | Eligible bindings |
| --- | --- | --- |
| `_repoint_loaded_aliases` | `startswith("vllm")` | Same-name references to the original module attribute or replaced submodule object, including MetaX modules. |
| Public-export scan in `_patch_module` | `startswith("vllm.")` | Same-name public callable/module exports from a previously loaded replaced submodule. The original and replacement modules themselves are excluded. |

The second scan currently excludes `vllm_metax.*`. It records the original
caller exports on the replacement **module**, not on each replacement callable.

## Patch metadata

`PATCH_MARKER` is the constant string `"__vllm_metax_patch__"`. The installer
attaches an attribute with this name to the **replacement object**. Its value is
a dictionary from each fully qualified replaced binding to the original object:

```python
replacement.__vllm_metax_patch__ == {
    "vllm.source.target": original,
    "vllm.caller.target": original,
    "vllm_metax.caller.target": original,
}
```

These values are references to the actual originals, not copies or function
names. Multiple targets and automatically updated aliases accumulate in the
same dictionary. For class-method/property patches, the marker lives on the
underlying replacement function/getter and records the original descriptor.
For `replace_module=True`, it lives on the replacement module and records the
previous `sys.modules` entry; an unloaded or intentionally absent target is
represented by the private `_MISSING` sentinel.

The dictionary serves three purposes:

1. **Recover the original for delegation or inspection.** A module-function
   wrapper can call its captured original without looking up the already
   replaced export again:

   ```python
   from vllm_metax.patch.utils import PATCH_MARKER, patch

   TARGET = "vllm.source.target"

   @patch("vllm.source", "target")
   def replacement(*args, **kwargs):
       original = getattr(replacement, PATCH_MARKER)[TARGET]
       return original(*args, **kwargs)
   ```

2. **Reject duplicate/conflicting installation.** If the current target already
   carries patch metadata, installing another patch raises `RuntimeError`.
   Installing the identical replacement again is also rejected. Inspecting this
   marker does not require a separate global table of patched function names.
3. **Explain what was rebound.** Each defining export and synchronized alias
   has its own dictionary key, preserving the original for that location.

`__vllm_metax_patch_aliases__` is a separate internal **set of binding names**,
created by `_repoint_loaded_aliases`. It distinguishes aliases updated
automatically from targets already installed explicitly. This permits stacked
module-attribute decorators:

```python
@patch("vllm.caller", "target")    # Runs second: promote the automatic alias.
@patch("vllm.source", "target")    # Runs first: replace source and sync caller.
def replacement(...):
    ...
```

The outer decorator finds that the caller already points to this replacement
and is in the automatic-alias set. It removes that set entry, keeps the saved
original dictionary entry, and returns once. A further explicit installation
on the caller raises `RuntimeError`. Whole-submodule replacement and
class-attribute replacement do not use this promotion exception.

The markers do not trigger patch activation, wrap calls automatically, or
provide an unpatch API. Activation still happens when the patch module imports;
delegating to an original requires an explicit call such as the wrapper above.

## Common examples

### Submodule replacement

Use explicit module replacement for lazy imports such as the MACA allocator:

```python
from vllm_metax.device_allocator import cumem
from vllm_metax.patch import patch

patch("vllm.device_allocator.cumem", replace_module=True)(cumem)
```

This sets both `sys.modules["vllm.device_allocator.cumem"]` and the parent
package's `cumem` attribute. It checks that the target module exists without
executing its original code. Thus `importlib.import_module(...)` and
`from vllm.device_allocator.cumem import CuMemAllocator` reach MetaX even when
the upstream leaf has not been imported yet. Replacing only a package attribute
does not provide this behavior.

For a previously loaded module, same-name public class/function imports in loaded
`vllm.*` callers are updated by identity. Renamed exports, existing instances and
references inside containers are outside this scan; install allocator patches
before allocation begins. Repeated/conflicting module replacements raise
`RuntimeError`. The replacement must be a module object, and this mode accepts no
attribute argument. `allow_missing=True` is reserved for intentionally adding a
new module, not bypassing a misspelled existing target.

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

| Directory | Ownership |
| --- | --- |
| `models/<model>/` | Workarounds, compatibility features and optimizations whose behavior is specific to a model or model family. Currently: `deepseek_v4`, `joyai`, `kimi_k3`, `minimax_m3`, `qwen`, `telechat3`. |
| `bugfix/` | Shared correctness and integration fixes, including generic communication, tokenizer, quantization and Triton restrictions. |
| `enhancement/` | Shared MetaX compatibility features, backend dispatch and runtime options. |
| `enhancement/performance/` | Shared performance replacements: grouped routing, sampling and speculative-decode batch ordering. |
| `template/` | Inactive, ready-to-copy patch examples. |
| `utils.py` | The shared `@patch` implementation. |

Classify by behavior and affected consumers, not only the model that first
exposed the issue or the module containing the target. A Qwen-only branch in
the generic weight loader belongs in `models/qwen/`. Conversely, the PP
all-gather workaround affects every `GroupCoordinator`, so it belongs in
`bugfix/parallel_state.py` even though DeepSeek-V4 exposed the failure. Keep
shared tokenizer/config wrappers in `bugfix/` while they serve multiple models;
do not split or install the same target twice just to match model directories.

Performance is a subset of enhancement. Nesting it keeps shared optimizations
discoverable without making it compete with model ownership at the top level.
Keep model-specific optimizations in `models/<model>/`, and describe whether a
patch fixes correctness, adds compatibility or improves speed in its header.

### Activation and directory moves

Both `register_out_of_tree()` and `register_attn_backends()` call the guarded
`_ensure_global_patch()` in `vllm_metax/__init__.py`. It imports `bugfix`,
`enhancement`, then `models` once per process. `register_out_of_tree()` loads
the ordinary registries afterward.

`enhancement/__init__.py` imports `performance` last, after allocator,
communication, quantization and runner setup. The performance package imports
its siblings with `from . import ...`. Do not import it separately from the
plugin entry point or restore the retired `patch.performance` path.

These directories organize eager plugin activation; `models/` does not select
patches based on the model being served. Model-only behavior still needs an
appropriate target or runtime guard. After a move, check relative and absolute
imports, the parent activation chain, initialization order and the audit table.
Avoid compatibility aliases under retired paths: importing the same patch under
two names can attempt to install it twice.

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
