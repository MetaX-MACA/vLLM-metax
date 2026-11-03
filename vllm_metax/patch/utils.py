# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Provide the shared patch decorator used to install module and class
#     replacements, preserve descriptors, record original attributes and reject
#     duplicate patches. Rebind loaded vLLM aliases by identity so callers reach
#     the installed export. Explicit submodule replacement also redirects the
#     import cache and parent package. This is infrastructure, not an upstream bug fix.
#
# Affected versions: MetaX patch infrastructure with vLLM 0.30.0
#     (ced6857afa0e); module redirection reviewed on 0.32.0 (5df63a7203).
#
# Remove at: All MetaX runtime patches migrate to native upstream extension points or
#     another shared installer, leaving no callers of this utility.
# -----------------------------------------------------------------------------

r"""Shared helpers for installing vLLM-MetaX runtime patches.

Installation flow (also documented in patch/README.md)::

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

Module target state [A]::

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

Missing attributes raise AttributeError unless allow_missing=True. Submodule
replacement checks the unloaded target spec without executing the original leaf;
a missing module raises ModuleNotFoundError unless allow_missing=True. The generic
alias scan visits loaded sys.modules entries with names starting with "vllm",
including vllm_metax and other matching prefixes. Only same-name references to
the identical original object are rebound. The separate public-export scan for
a replaced submodule uses "vllm." instead and excludes MetaX callers. Class
aliases already share their owner, so class-attribute patches do not scan module
exports. Renamed references such as _original_target remain untouched; same-name
MetaX references are eligible for the generic scan.

The replacement object's __vllm_metax_patch__ dictionary maps each replaced
binding's fully qualified name to its original object. It supports explicit
delegation, inspection and duplicate-patch rejection; it does not wrap calls or
restore bindings automatically. Class patches attach it to the replacement
function/getter and save the original descriptor. Module redirection attaches
it to the replacement module; an unloaded original is the private _MISSING
sentinel. The separate __vllm_metax_patch_aliases__ set tracks generic aliases
updated automatically. A stacked module-attribute decorator can promote one
such alias to an explicit target once without overwriting its saved original.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from collections.abc import Callable
from types import ModuleType
from typing import Any, TypeVar, cast

PatchTarget = TypeVar("PatchTarget")

# Stored on the replacement: {fully qualified binding: original object}.
# Used for delegation/inspection and to reject another patch on that target.
# Class patches record descriptors on the underlying function/getter; module
# redirection records cached modules (or _MISSING) on the replacement module.
PATCH_MARKER = "__vllm_metax_patch__"
# Track automatic bindings separately so a stacked decorator can explicitly
# name one once without losing its original object or accepting duplicate targets.
_ALIAS_PATCH_MARKER = "__vllm_metax_patch_aliases__"
_MISSING = object()


def _repoint_loaded_aliases(
    attribute_name: str, original: Any, replacement: Any
) -> None:
    """Update loaded modules with names starting with ``vllm``, including MetaX.

    Future imports resolve the replacement through its defining module. Inspect
    existing module dictionaries directly so lazy attribute hooks are not invoked;
    identity checks leave unrelated exports with the same name untouched.

    A caller loaded via ``from vllm.some_module import target`` before patching
    keeps the old function object even after ``some_module.target`` is replaced.
    Module-attribute installation calls this helper automatically. Alias updates
    record their original objects without recursively invoking the decorator.
    For example, all_reduce_utils.CudaRTLibrary must follow the MACA replacement even
    when all_reduce_utils was imported first during model-module initialization.
    """
    for module_name, module in list(sys.modules.items()):
        if module is None or not module_name.startswith("vllm"):
            continue
        if vars(module).get(attribute_name, _MISSING) is original:
            full_target_name = f"{module_name}.{attribute_name}"
            _record_original_attribute(replacement, full_target_name, original)
            alias_targets = getattr(replacement, _ALIAS_PATCH_MARKER, None)
            if alias_targets is None:
                alias_targets = set()
                setattr(replacement, _ALIAS_PATCH_MARKER, alias_targets)
            alias_targets.add(full_target_name)
            setattr(module, attribute_name, replacement)


def _record_original_attribute(
    replacement_implementation: Any,
    full_target_name: str,
    original_attribute: Any,
) -> None:
    """Add one replaced attribute to a replacement's single patch marker."""
    original_attributes = getattr(replacement_implementation, PATCH_MARKER, None)
    if original_attributes is None:
        original_attributes = {}
    elif not isinstance(original_attributes, dict):
        raise TypeError(f"Invalid patch metadata on {replacement_implementation!r}")

    original_attributes[full_target_name] = original_attribute
    try:
        setattr(replacement_implementation, PATCH_MARKER, original_attributes)
    except (AttributeError, TypeError) as exc:
        raise TypeError(
            f"Replacement for {full_target_name} cannot store patch metadata"
        ) from exc


def _read_original_attribute(
    owner: Any,
    attribute_name: str,
    full_target_name: str,
    allow_missing: bool,
) -> Any:
    """Read an attribute without invoking descriptors or user lookup hooks."""
    try:
        return inspect.getattr_static(owner, attribute_name)
    except AttributeError as exc:
        if allow_missing:
            return _MISSING
        raise AttributeError(f"{full_target_name!r} does not exist") from exc


def _raise_if_already_patched(
    original_implementation: Any,
    replacement_implementation: Any,
    full_target_name: str,
) -> None:
    """Reject both reapplying one object and installing a second patch."""
    if (
        original_implementation is replacement_implementation
        or getattr(original_implementation, PATCH_MARKER, None) is not None
    ):
        raise RuntimeError(f"{full_target_name} is already patched")


def _patch_module_attribute(
    target_module_path: str,
    target_attribute_name: str | None = None,
    *,
    allow_missing: bool = False,
) -> Callable[[PatchTarget], PatchTarget]:
    """Replace a module-level function, class, Triton kernel, or other object.

    This path intentionally has no class-descriptor handling. The object returned
    by the inner decorators is installed in the module unchanged. This is why
    ``@patch`` must be outside ``@triton.jit`` and ``@triton.autotune``.
    """
    if not target_module_path:
        raise ValueError("target_module_path must not be empty")

    def install_module_patch(replacement_attribute: PatchTarget) -> PatchTarget:
        resolved_attribute_name = target_attribute_name or getattr(
            replacement_attribute, "__name__", ""
        )
        if not resolved_attribute_name:
            raise ValueError(
                "target_attribute_name is required for unnamed replacement objects"
            )

        target_module = importlib.import_module(target_module_path)
        full_target_name = f"{target_module_path}.{resolved_attribute_name}"
        original_attribute = _read_original_attribute(
            target_module,
            resolved_attribute_name,
            full_target_name,
            allow_missing,
        )

        # An inner decorator can already have updated this target as an alias.
        # Promote it to an explicit target once, retaining its original metadata.
        # Repeating an explicit target still fails the duplicate check below.
        alias_targets: set[str] = getattr(
            replacement_attribute, _ALIAS_PATCH_MARKER, set()
        )
        if (
            original_attribute is replacement_attribute
            and full_target_name in alias_targets
        ):
            alias_targets.remove(full_target_name)
            return replacement_attribute

        if original_attribute is not _MISSING:
            _raise_if_already_patched(
                original_attribute,
                replacement_attribute,
                full_target_name,
            )

        _record_original_attribute(
            replacement_attribute,
            full_target_name,
            original_attribute,
        )
        setattr(target_module, resolved_attribute_name, replacement_attribute)
        if original_attribute is not _MISSING:
            _repoint_loaded_aliases(
                resolved_attribute_name, original_attribute, replacement_attribute
            )
        return replacement_attribute

    return install_module_patch


def _patch_module(
    target_module_path: str, *, allow_missing: bool = False
) -> Callable[[ModuleType], ModuleType]:
    """Redirect a submodule without executing its original implementation."""
    parent_path, _, attribute_name = target_module_path.rpartition(".")
    if not parent_path or not attribute_name:
        raise ValueError("Module replacement requires a qualified submodule path")

    def install_module_patch(replacement: ModuleType) -> ModuleType:
        if not isinstance(replacement, ModuleType):
            raise TypeError("Module replacement requires a module object")
        parent = importlib.import_module(parent_path)
        original = sys.modules.get(target_module_path, _MISSING)
        original_attribute = vars(parent).get(attribute_name, _MISSING)
        for previous in (original, original_attribute):
            if previous is not _MISSING:
                _raise_if_already_patched(previous, replacement, target_module_path)
        # NOTE(metax): Inspect the spec instead of importing the CUDA allocator.
        if (
            original is _MISSING
            and not allow_missing
            and importlib.util.find_spec(target_module_path) is None
        ):
            raise ModuleNotFoundError(
                f"{target_module_path!r} does not exist",
                name=target_module_path,
            )

        _record_original_attribute(replacement, target_module_path, original)
        sys.modules[target_module_path] = replacement
        setattr(parent, attribute_name, replacement)

        if isinstance(original, ModuleType):
            _repoint_loaded_aliases(attribute_name, original, replacement)
            # NOTE(metax): Rebind preloaded allocator class/function imports too.
            # Constants and renamed exports cannot be attributed by identity.
            for name, value in list(vars(original).items()):
                updated = vars(replacement).get(name, _MISSING)
                if (
                    name.startswith("_")
                    or not (callable(value) or isinstance(value, ModuleType))
                    or updated is _MISSING
                    or updated is value
                ):
                    continue
                for module_name, module in list(sys.modules.items()):
                    if (
                        module is None
                        or not module_name.startswith("vllm.")
                        or module is original
                        or module is replacement
                    ):
                        continue
                    if vars(module).get(name, _MISSING) is value:
                        _record_original_attribute(
                            replacement, f"{module_name}.{name}", value
                        )
                        setattr(module, name, updated)
        return replacement

    return install_module_patch


def _unwrap_class_descriptor(attribute: Any) -> Any:
    """Return the implementation stored inside a supported class descriptor."""
    if isinstance(attribute, (staticmethod, classmethod)):
        return attribute.__func__
    if isinstance(attribute, property):
        return attribute.fget
    return attribute


def _build_class_attribute(
    original_attribute: Any,
    replacement_attribute: Any,
    replacement_implementation: Any,
    full_target_name: str,
) -> Any:
    """Build a class attribute with the original descriptor semantics."""
    if original_attribute is _MISSING:
        return replacement_attribute
    if isinstance(original_attribute, staticmethod):
        return staticmethod(replacement_implementation)
    if isinstance(original_attribute, classmethod):
        return classmethod(replacement_implementation)
    if isinstance(original_attribute, property):
        return property(
            replacement_implementation,
            original_attribute.fset,
            original_attribute.fdel,
            original_attribute.__doc__,
        )
    if isinstance(replacement_attribute, (staticmethod, classmethod, property)):
        raise TypeError(f"Replacement descriptor does not match {full_target_name}")
    return replacement_implementation


def _resolve_target_class(target_module: Any, target_class_path: str) -> type[Any]:
    """Resolve a dotted class path without importing it in the patch module."""
    target_class = target_module
    for class_name in target_class_path.split("."):
        try:
            target_class = getattr(target_class, class_name)
        except AttributeError as exc:
            raise AttributeError(
                f"Cannot resolve target class {target_class_path!r}"
            ) from exc
    if not isinstance(target_class, type):
        raise TypeError(f"{target_class_path!r} does not resolve to a class")
    return target_class


def _patch_class_method(
    target_module_path: str,
    target_attribute_path: str,
    *,
    allow_missing: bool = False,
) -> Callable[[PatchTarget], PatchTarget]:
    """Replace a class method or property through a dotted attribute path.

    Import and class resolution stay inside this path, so patch modules do not
    need to import a class only to identify the patch target.
    """
    target_class_path, separator, target_attribute_name = (
        target_attribute_path.rpartition(".")
    )
    if not separator or not target_class_path or not target_attribute_name:
        raise ValueError("Class patches require a path such as 'TargetClass.method'")

    def install_class_patch(replacement_attribute: PatchTarget) -> PatchTarget:
        replacement_implementation = _unwrap_class_descriptor(replacement_attribute)
        target_module = importlib.import_module(target_module_path)
        target_class = _resolve_target_class(target_module, target_class_path)
        full_target_name = f"{target_module_path}.{target_attribute_path}"
        original_attribute = _read_original_attribute(
            target_class,
            target_attribute_name,
            full_target_name,
            allow_missing,
        )

        if original_attribute is not _MISSING:
            original_implementation = _unwrap_class_descriptor(original_attribute)
            _raise_if_already_patched(
                original_implementation,
                replacement_implementation,
                full_target_name,
            )

        replacement_class_attribute = _build_class_attribute(
            original_attribute,
            replacement_attribute,
            replacement_implementation,
            full_target_name,
        )
        _record_original_attribute(
            replacement_implementation,
            full_target_name,
            original_attribute,
        )
        setattr(
            target_class,
            target_attribute_name,
            replacement_class_attribute,
        )
        return cast(PatchTarget, replacement_class_attribute)

    return install_class_patch


def patch(
    target_module_path: str,
    target_attribute_name: str | None = None,
    *,
    allow_missing: bool = False,
    replace_module: bool = False,
) -> Callable[[PatchTarget], PatchTarget]:
    """Create a patch decorator for an attribute or an entire submodule.

    ``replace_module=True`` redirects the fully qualified module path in
    ``sys.modules`` and its parent package without importing the original leaf.
    Existing same-name vLLM class/function imports are synchronized; renamed
    exports and existing instances remain outside this replacement.

    ``target_attribute_name`` selects one of two independent implementations:

    - a direct or inferred name uses the module-attribute path, which installs
      the replacement unchanged and supports Triton JIT/autotune objects. It
      also updates same-name exports in loaded modules with names starting with
      ``vllm`` (including ``vllm_metax``) that still reference the original object;
    - a dotted path such as ``"TargetClass.method"`` uses the class-method path,
      which resolves the class internally and preserves method descriptors.

    Automatically updated aliases may also be explicit stacked targets. Their
    first explicit installation preserves the already-recorded original object;
    a second explicit installation raises ``RuntimeError`` as usual. Class
    aliases share their owner and need no module-export synchronization when
    patching a method or property on that class.
    """
    if not isinstance(target_module_path, str) or not target_module_path:
        raise TypeError("target_module_path must be a non-empty module import path")
    if replace_module:
        if target_attribute_name is not None:
            raise ValueError("Module replacement does not accept an attribute name")
        return cast(
            Callable[[PatchTarget], PatchTarget],
            _patch_module(target_module_path, allow_missing=allow_missing),
        )
    if target_attribute_name is not None and "." in target_attribute_name:
        return _patch_class_method(
            target_module_path,
            target_attribute_name,
            allow_missing=allow_missing,
        )
    return _patch_module_attribute(
        target_module_path,
        target_attribute_name,
        allow_missing=allow_missing,
    )
