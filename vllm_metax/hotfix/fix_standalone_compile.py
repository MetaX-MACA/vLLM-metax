# SPDX-License-Identifier: Apache-2.0
import torch

# from typing import TYPE_CHECKING
from torch._functorch._aot_autograd.schemas import AOTConfig
from torch._functorch._aot_autograd.autograd_cache import (
    check_cacheable,
    AOTAutogradCacheDetails,
    AOTAutogradCachePickler,
)
from torch._inductor.compile_fx import _CompileFxKwargs


def autograd_cache_key(
    gm: torch.fx.GraphModule,
    example_inputs,
    config: AOTConfig,
    fx_config: _CompileFxKwargs,
    # TODO: add args and parameters
) -> tuple[str, list[str]]:
    """
    Generate a unique hash of the FX graph for caching.
    """
    check_cacheable(gm)

    details = AOTAutogradCacheDetails(gm, example_inputs, config, fx_config)
    pickler = AOTAutogradCachePickler(gm)
    # The prefix distinguishes among the other kinds of objects we cache
    key = "a" + pickler.get_hash(details)
    debug_lines = pickler.debug_lines(details)
    return key, debug_lines


import torch._functorch._aot_autograd.autograd_cache

torch._functorch._aot_autograd.autograd_cache.autograd_cache_key = autograd_cache_key
