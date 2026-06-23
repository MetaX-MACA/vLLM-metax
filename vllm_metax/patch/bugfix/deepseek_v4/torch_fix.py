# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# -----------------------------------------------
# Note: Work around PyTorch compilation backend regressions affecting MetaX
#       standalone compilation and artifact caching.
#
# Affected versions: v0.21.0-dsv4
# Remove at: after PyTorch 2.10+metax and the related vLLM compilation fixes
#            are available.
# -----------------------------------------------


from vllm.compilation.backends import logger
from vllm.compilation.counter import compilation_counter
from vllm.compilation.compiler_interface import (
    InductorAdaptor,
    is_compile_cache_enabled,
)

import time

import torch.fx as fx

from typing import Literal, Sequence, Any
from vllm.config import CompilationConfig
from vllm.config.utils import Range
from vllm.tracing import instrument

import contextlib
import copy
import os
from collections.abc import Callable
from unittest.mock import patch

import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.config.utils import Range
from vllm.env_override import _apply_constrain_to_fx_strides_patch
from vllm.utils.torch_utils import is_torch_equal_or_newer

from vllm.compilation.compiler_interface import (
    InductorStandaloneAdaptor,
    set_inductor_config,
    set_functorch_config,
)


# --------------------------------------------------
# This hotfix is to revert:
#   https://github.com/vllm-project/vllm/pull/36093
# which causes a attribute error on torch2.8+metax.
#
# TODO(hank): Remove this once pytorch 2.10+metax is released.
# --------------------------------------------------
def inductor_standalone_compile(
    self,
    graph: fx.GraphModule,
    example_inputs: list[Any],
    compiler_config: dict[str, Any],
    compile_range: Range,
    key: str | None = None,
) -> tuple[Callable[..., Any] | None, Any | None]:
    _apply_constrain_to_fx_strides_patch()
    compilation_counter.num_inductor_compiles += 1
    current_config = {}
    if compiler_config is not None:
        current_config.update(compiler_config)
    set_inductor_config(current_config, compile_range)
    set_functorch_config()

    if compile_range.is_single_size():
        dynamic_shapes = "from_example_inputs"
    else:
        dynamic_shapes = "from_graph"

    from torch._inductor import standalone_compile

    supports_aot = is_torch_equal_or_newer("2.10.0")

    if not supports_aot and envs.VLLM_USE_MEGA_AOT_ARTIFACT:
        logger.error(
            "CRITICAL: VLLM_USE_MEGA_AOT_ARTIFACT "
            "is enabled but PyTorch version does not support 'aot' "
            "parameter in standalone_compile. This requires PyTorch "
            "2.10.0+. Falling back to non-AOT mode."
        )

    compile_kwargs = {
        "dynamic_shapes": dynamic_shapes,
        "options": {
            "config_patches": current_config,
        },
    }

    use_aot: bool = supports_aot and envs.VLLM_USE_MEGA_AOT_ARTIFACT
    # only add 'aot' parameter if both supported and enabled...
    # this will set bundled_autograd_cache
    # https://github.com/pytorch/pytorch/blob/9bbc5b2905c260adf41bc866a732f9c121a2828a/torch/_inductor/standalone_compile.py#L359 # noqa
    if use_aot:
        compile_kwargs["aot"] = True  # type: ignore[assignment]

    # Inductor's pre-grad passes don't do anything for vLLM.
    # The pre-grad passes get run even on cache-hit and negatively impact
    # vllm cold compile times by O(1s)
    # Can remove this after the following issue gets fixed
    # https://github.com/pytorch/pytorch/issues/174502
    if envs.VLLM_ENABLE_PREGRAD_PASSES:
        pregrad_ctx: Any = contextlib.nullcontext()
    else:
        pregrad_ctx = patch(
            "torch._inductor.compile_fx._recursive_pre_grad_passes",
            lambda gm, _: gm,
        )
    # When inputs are FakeTensors (from create_concrete_args),
    # standalone_compile("from_example_inputs") would normally create
    # a fresh FakeTensorMode, causing a mode mismatch assertion.
    # Patch FakeTensorMode in standalone_compile so it reuses the
    # mode already attached to our FakeTensors. This gives us both
    # ignore_shape_env=True (from "from_example_inputs") and mode
    # consistency (from reusing our mode).
    # Can remove this after the following issue gets fixed:
    # https://github.com/pytorch/pytorch/issues/176562
    from torch._subclasses.fake_tensor import FakeTensor

    input_fake_mode = None
    for x in example_inputs:
        if isinstance(x, FakeTensor):
            input_fake_mode = x.fake_mode
            break

    if input_fake_mode is not None:
        # Use patch.object on the actual module from sys.modules
        # because in Python <=3.10 the string-based patch() resolves
        # torch._inductor.standalone_compile to the wrapper function
        # (defined in __init__.py) instead of the module.
        import sys

        fake_mode_ctx: Any = patch.object(
            sys.modules["torch._inductor.standalone_compile"],
            "FakeTensorMode",
            lambda *a, **kw: input_fake_mode,
        )
    else:
        fake_mode_ctx = contextlib.nullcontext()

    with pregrad_ctx, fake_mode_ctx:
        compiled_graph = standalone_compile(graph, example_inputs, **compile_kwargs)

    if use_aot:
        from torch._inductor.standalone_compile import AOTCompiledArtifact

        assert isinstance(compiled_graph, AOTCompiledArtifact)
        assert hasattr(compiled_graph, "serialize")
        # just return the compiled graph and a key
        # since we can serialize the bytes using to_bytes
        # and reload it using the key when reading
        return compiled_graph, None

    # Save the compiled artifact to disk in the specified path
    assert key is not None
    path = os.path.join(self.cache_dir, key)

    def is_saveable_2_10(compiled_artifact):
        # can just use compiled_artifact.is_saveable in 2.11
        if compiled_artifact._artifacts is None:
            return False
        _, cache_info = compiled_artifact._artifacts
        return len(cache_info.aot_autograd_artifacts) == 1

    if is_compile_cache_enabled(compiler_config):
        if not is_saveable_2_10(compiled_graph):
            raise RuntimeError(
                "The compiled artifact is not serializable. This usually means "
                "that the model code has something that is not serializable "
                "by torch.compile in it. You can fix this by either "
                "figuring out what is not serializable and rewriting it, "
                "filing a bug report, "
                "or suppressing this error by "
                "disabling vLLM's compilation cache via "
                "VLLM_DISABLE_COMPILE_CACHE=1 "
                "(this will greatly increase vLLM server warm start times)."
            )
        compiled_graph.save(path=path, format=self.save_format)
        compilation_counter.num_compiled_artifacts_saved += 1
    return compiled_graph, (key, path)


InductorStandaloneAdaptor.compile = inductor_standalone_compile


# --------------------------------------------------
# This hotfix is to revert:
#   https://github.com/vllm-project/vllm/pull/34003
# which causes a attribute error on torch2.8+metax.
#
# TODO(hank): Remove this once vllm-metax v0.17.0 is released.
# --------------------------------------------------
@instrument(span_name="Compile graph")
def compile(
    self,
    graph: fx.GraphModule,
    example_inputs: list[Any],
    additional_inductor_config: dict[str, Any],
    compilation_config: CompilationConfig,
    compile_range: Range,
    graph_index: int = 0,
    num_graphs: int = 1,
    is_encoder: bool = False,
) -> Any:
    if graph_index == 0:
        # before compiling the first graph, record the start time
        global compilation_start_time
        compilation_start_time = time.perf_counter()

    compilation_counter.num_backend_compilations += 1

    compiled_graph = None

    # try to load from the cache
    compiled_graph = self.load(graph, example_inputs, graph_index, compile_range)
    if compiled_graph is not None:
        if graph_index == num_graphs - 1:
            # after loading the last graph for this shape, record the time.
            # there can be multiple graphs due to piecewise compilation.
            elapsed = time.perf_counter() - compilation_start_time
            if is_encoder:
                compilation_config.encoder_compilation_time += elapsed
            else:
                compilation_config.compilation_time += elapsed
            logger.info_once(
                "Directly load the compiled graph(s) for compile range %s "
                "from the cache, took %.3f s",
                str(compile_range),
                elapsed,
                scope="local",
            )
        return compiled_graph

    # no compiler cached the graph, or the cache is disabled,
    # we need to compile it
    if isinstance(self.compiler, InductorAdaptor):
        # Let compile_fx generate a key for us
        maybe_key = None
    else:
        maybe_key = "artifact_compile_range_"
        maybe_key += f"{compile_range.start}_{compile_range.end}"
        maybe_key += f"_subgraph_{graph_index}"
    with self.compile_context(compile_range):
        cache_key = None
        compiled_graph, handle = self.compiler.compile(
            graph,
            example_inputs,
            additional_inductor_config,
            compile_range,
            maybe_key,
        )
        assert compiled_graph is not None, "Failed to compile the graph"

    # store the artifact in the cache
    if is_compile_cache_enabled(additional_inductor_config) and handle is not None:
        self.cache[(compile_range, graph_index, self.compiler.name)] = {
            "graph_handle": handle,
            "cache_key": cache_key,
        }
        compilation_counter.num_cache_entries_updated += 1
        self.is_cache_updated = True
        if graph_index == 0:
            # adds some info logging for the first graph
            logger.info_once(
                "Cache the graph of compile range %s for later use",
                str(compile_range),
            )
        logger.debug_once(
            "Store the %s-th graph for compile range%s from %s via handle %s",
            graph_index,
            str(compile_range),
            self.compiler.name,
            handle,
            scope="local",
        )

    # after compiling the last graph, record the end time
    if graph_index == num_graphs - 1:
        elapsed = time.perf_counter() - compilation_start_time
        if is_encoder:
            compilation_config.encoder_compilation_time += elapsed
        else:
            compilation_config.compilation_time += elapsed
        logger.info_once(
            "Compiling a graph for compile range %s takes %.2f s",
            str(compile_range),
            elapsed,
            scope="local",
        )

    return compiled_graph


from vllm.compilation.backends import CompilerManager

CompilerManager.compile = compile

# -----------------------------------------------
# Note: Hotfix the standalone compilation backend for torch2.8+metax.
#
# Affected versions: v0.21.0-dsv4
# Remove at: after the underlying torch issue is resolved.
# -----------------------------------------------

import torch

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


# --------------------------------------------------------------------------------
# Hotfix for enabling torch2.8+metax to work with accelerator API in torch 2.9+ !!!
torch.accelerator.empty_cache = torch.cuda.empty_cache
torch.accelerator.memory_stats = torch.cuda.memory_stats
torch.accelerator.memory_reserved = torch.cuda.memory_reserved
torch.accelerator.reset_peak_memory_stats = torch.cuda.reset_peak_memory_stats
torch.accelerator.max_memory_allocated = torch.cuda.max_memory_allocated
