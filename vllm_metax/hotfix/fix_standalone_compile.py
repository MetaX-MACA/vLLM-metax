# SPDX-License-Identifier: Apache-2.0

# -----------------------------------------------------
# Note: This is a hotfix for torch2.8+metax to make the
#       standalone compilation backend work.
#
# TODO(hank): Remove this once the torch issue is resolved.
# _____________________________________________________

import torch
import copy

from torch._functorch._aot_autograd.schemas import AOTConfig
from torch._functorch._aot_autograd.autograd_cache import (
    check_cacheable,
    AOTAutogradCacheDetails,
    AOTAutogradCachePickler,
)
from torch._inductor.compile_fx import _CompileFxKwargs

from torch._subclasses import FakeTensorMode
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch._inductor.standalone_compile import CompiledArtifact, config, log
from typing import Literal, Sequence, Any
from torch._inductor.utils import InputType


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


def standalone_compile_impl(
    gm: torch.fx.GraphModule,
    example_inputs: Sequence[InputType],
    *,
    dynamic_shapes: Any,
    options: Any,
) -> CompiledArtifact:
    from torch.compiler._cache import CacheArtifactManager

    from torch._inductor.compile_fx import compile_fx

    ignore_shape_env = False
    if dynamic_shapes == "from_example_inputs":
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        # tells compile_fx to ignore the shape_envs on the ambient context
        # and the graph_module.
        ignore_shape_env = True
    elif dynamic_shapes == "from_tracing_context":
        # Reuse fake_mode from the TracingContext.
        # NB: The TracingContext only exists if we're currently in a torch.compile backend.
        context = torch._guards.TracingContext.get()
        fake_mode = context.fake_mode
    elif dynamic_shapes == "from_graph":
        fake_mode = FakeTensorMode(shape_env=ShapeEnv())
        # Strategy: find a FakeTensor in the graph output, grab its FakeTensorMode.
        # The graph passed to standalone_compile must be an Inductor-approved graph,
        # which means that there is at least one Tensor output and the output node
        # contains a flat list of Tensors.
        last_node = next(iter(reversed(gm.graph.nodes)))
        assert last_node.op == "output"
        assert len(last_node.args) == 1
        # ============== modification starts here =================
        out = last_node.args[0]

        def _iter_fx_nodes(x):
            # FX output can be a Node, or nested (tuple/list/dict) of Nodes.
            if isinstance(x, torch.fx.Node):
                yield x
            elif isinstance(x, (tuple, list)):
                for y in x:
                    yield from _iter_fx_nodes(y)
            elif isinstance(x, dict):
                for y in x.values():
                    yield from _iter_fx_nodes(y)
            else:
                return

        for node in _iter_fx_nodes(out):
            # ================ modification ends here =================
            if "example_value" in node.meta:
                maybe_tensor = node.meta["example_value"]
                if isinstance(maybe_tensor, torch._subclasses.fake_tensor.FakeTensor):
                    fake_mode = maybe_tensor.fake_mode
    else:
        raise ValueError(
            f"standalone_compile got unsupported `dynamic_shapes` value: dynamic_shapes={dynamic_shapes}."
        )

    context = torch._guards.TracingContext(fake_mode)
    with (
        torch._guards.tracing(context),
        CacheArtifactManager.with_fresh_cache(),
        config.patch("triton.autotune_at_compile_time", True),
    ):
        # compile_fx can mutate gm
        gm = copy.deepcopy(gm)
        compiled_fn = compile_fx(
            gm, example_inputs, ignore_shape_env=ignore_shape_env, **options
        )
        assert callable(compiled_fn)

        artifacts = torch.compiler.save_cache_artifacts()
        if artifacts is None:
            log.warning(
                "standalone_compile artifact generation failed, cannot save. "
                "Run with TORCH_LOGS=+torch._inductor.codecache to identify the problem"
            )

    return CompiledArtifact(compiled_fn, artifacts)


def standalone_compile(
    gm: torch.fx.GraphModule,
    example_inputs: list[InputType],
    *,
    dynamic_shapes: Literal[
        "from_example_inputs", "from_tracing_context", "from_graph"
    ] = "from_graph",
    options: dict[str, Any] | None = None,
) -> CompiledArtifact:
    """
    Precompilation API for inductor.

    .. code-block:: python

        compiled_artifact = torch._inductor.standalone_compile(gm, args)
        compiled_artifact.save(path=path, format="binary")

        # Later on a new process
        loaded = torch._inductor.CompiledArtifact.load(path=path, format="binary")
        compiled_out = loaded(*args)

    Args:
        gm: Graph Module
        example_inputs: Inputs for the graph module
        dynamic_shapes: If "from_graph" (default), we will use the dynamic
            shapes in the passed-in graph module.
            If "from_tracing_context", we use the dynamic shape info in the
            ambient tracing context.
            If "from_example_inputs", we will specialize the graph on the
            example_inputs.
        options: Inductor compilation options

    Returns:
        CompiledArtifact that can be saved to disk or invoked directly.
    """

    options = options if options else {}
    return standalone_compile_impl(
        gm, example_inputs, dynamic_shapes=dynamic_shapes, options=options
    )


import torch._inductor

torch._inductor.standalone_compile = standalone_compile

import torch._functorch._aot_autograd.autograd_cache

torch._functorch._aot_autograd.autograd_cache.autograd_cache_key = autograd_cache_key


# -------------------------------------------------------------------
# Hotfix for enabling torch2.9 TF32 support in torch 2.8+metax !!!
#
# We recommend enabling TF32 tensor cores for matrix multiplications with
# torch.backends.cuda.matmul.fp32_precision = "tf32"
# (`torch.backends.cuda.matmul.allow_tf32 = True is going to be deprecated)
#
# https://docs.pytorch.org/docs/stable/notes/cuda.html


# Commenting out __getattr__ since we don't need patching it for now.
"""
def __getattr__(self, name):
    if name == "allow_tf32":
        return torch._C._get_cublas_allow_tf32()
    elif name == "allow_fp16_reduced_precision_reduction":
        return torch._C._get_cublas_allow_fp16_reduced_precision_reduction()
    elif name == "allow_bf16_reduced_precision_reduction":
        return torch._C._get_cublas_allow_bf16_reduced_precision_reduction()
    elif name == "allow_fp16_accumulation":
        return torch._C._get_cublas_allow_fp16_accumulation()
    elif name == "fp32_precision":
        return "tf32" if torch._C._get_cublas_allow_tf32() else "ieee"
    raise AttributeError("Unknown attribute " + name)
"""


def __setattr__(self, name, value):
    if name == "allow_tf32":
        return torch._C._set_cublas_allow_tf32(value)
    elif name == "allow_fp16_reduced_precision_reduction":
        return torch._C._set_cublas_allow_fp16_reduced_precision_reduction(value)
    elif name == "allow_bf16_reduced_precision_reduction":
        return torch._C._set_cublas_allow_bf16_reduced_precision_reduction(value)
    elif name == "allow_fp16_accumulation":
        return torch._C._set_cublas_allow_fp16_accumulation(value)
    elif name == "fp32_precision":
        return torch._C._set_cublas_allow_tf32(value == "tf32")
    raise AttributeError("Unknown attribute " + name)


from torch.backends.cuda import matmul

matmul.__class__.__setattr__ = __setattr__
