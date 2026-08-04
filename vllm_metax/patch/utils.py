# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Shared helpers for installing vLLM-MetaX runtime patches."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Callable
from typing import Any, TypeVar, cast

PatchTarget = TypeVar("PatchTarget")

# The marker identifies an installed patch and maps each target name to the
# original attribute that was replaced there.
PATCH_MARKER = "__vllm_metax_patch__"
_MISSING = object()


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
        return replacement_attribute

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
) -> Callable[[PatchTarget], PatchTarget]:
    """Create a patch decorator for a module attribute or class attribute.

    ``target_attribute_name`` selects one of two independent implementations:

    - a direct or inferred name uses the module-attribute path, which installs
      the replacement unchanged and supports Triton JIT/autotune objects;
    - a dotted path such as ``"TargetClass.method"`` uses the class-method path,
      which resolves the class internally and preserves method descriptors.
    """
    if not isinstance(target_module_path, str) or not target_module_path:
        raise TypeError("target_module_path must be a non-empty module import path")
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
