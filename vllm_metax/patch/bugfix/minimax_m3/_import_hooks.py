# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# -----------------------------------------------
# Note: Shared "patch on first import" helper for the minimax_m3 bugfixes.
#
# Importing anything under `vllm.models.minimax_m3` forces Python to first
# execute `vllm/models/minimax_m3/__init__.py`, which unconditionally does
# `from .nvidia.model import (...)` on non-ROCm platforms (MACA included) --
# which transitively imports flashinfer -> fla, and fla's own
# `TileLangBackend` registration ends up dlopen-ing `tilelang`'s bundled
# libraries as a side effect. On MetaX, `tilelang`'s bundled
# `libcudart_stub.so` happens to collide with vllm_metax's own
# `CudaRTLibrary`'s naive `/proc/self/maps` substring search for "libcudart"
# (see `vllm.distributed.device_communicators.cuda_wrapper.CudaRTLibrary`),
# which is invoked once, early and unconditionally, from
# `vllm_metax/patch/model_executor/layers/lamport_workspace.py` during
# general-plugin registration. If that resolution runs *before* anything has
# loaded the tilelang stub, it correctly finds and caches MetaX's real
# `libmcruntime.so` (the result is cached permanently); if `tilelang` is
# already loaded by that point, it incorrectly (and just as permanently)
# resolves to the broken stub instead, crashing with `AttributeError: ...
# undefined symbol: mcSetDevice`.
#
# Eagerly importing `vllm.models.minimax_m3.common.ops.index_topk` /
# `.sparse_attn` at the top of these bugfix modules -- needed to reuse the
# original kernels -- forces that whole chain (and thus `tilelang`) to load
# *during general-plugin registration*, i.e. before
# `lamport_workspace.py` has necessarily had its turn, creating exactly this
# race. `on_first_import` defers the actual patch application until the
# target module is imported by *someone*, whenever that naturally happens
# (typically much later, during real model loading -- well after general
# plugins, and thus `lamport_workspace.py`, have already run) instead of
# forcing it here.
#
# Affected versions: v0.24.0
# -----------------------------------------------

import importlib.util
import sys


def on_first_import(module_name, callback):
    """Call ``callback(module)`` right after ``module_name`` is first fully
    imported by *anyone*, without forcing that import to happen now.

    If the module is already imported, calls back immediately
    (synchronously, right here). Otherwise installs a one-shot
    ``sys.meta_path`` hook that fires the callback the moment the module
    finishes executing -- *before* control returns to whichever `import`
    statement triggered it -- guaranteeing every subsequent
    ``from module_name import x`` (including the very one that triggered
    the import) sees the patched attributes, no matter who imports it first.
    """
    existing = sys.modules.get(module_name)
    if existing is not None:
        callback(existing)
        return

    class _OnImportFinder:
        def find_spec(self, fullname, path, target=None):
            if fullname != module_name:
                return None
            # One-shot: stop intercepting and delegate to the rest of the
            # maca path (with ourselves removed, to avoid recursing back
            # into this very function) to find the real spec.
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
            if spec is None or spec.loader is None:
                return spec
            original_exec_module = spec.loader.exec_module

            def exec_module(module):
                original_exec_module(module)
                callback(module)

            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _OnImportFinder())
