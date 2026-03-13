# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# --------------------------------------------------
# This hotfix is to revert:
#   https://github.com/vllm-project/vllm/pull/34003
# which causes a attribute error on torch2.8+metax.
# --------------------------------------------------

from vllm.compilation.backends import logger
from vllm.compilation.counter import compilation_counter
from vllm.compilation.compiler_interface import (
    CompilerInterface,
    EagerAdaptor,
    InductorAdaptor,
    InductorStandaloneAdaptor,
    is_compile_cache_enabled,
)

import ast
import contextvars
import dataclasses
import hashlib
import json
import operator
import os
import pprint
import time

import torch
import torch.fx as fx

from typing import Any
import vllm.envs as envs
from vllm.config import CompilationConfig, CUDAGraphMode, VllmConfig
from vllm.config.compilation import DynamicShapesType
from vllm.config.utils import Range, hash_factors
from vllm.logger import init_logger
from vllm.logging_utils import lazy
from vllm.platforms import current_platform
from vllm.tracing import instrument, instrument_manual
from vllm.utils.import_utils import resolve_obj_by_qualname


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
        self.cache[(compile_range, graph_index, self.compiler.name)] = handle
        compilation_counter.num_cache_entries_updated += 1
        self.is_cache_updated = True
        if graph_index == 0:
            # adds some info logging for the first graph
            logger.info_once(
                "Cache the graph of compile range %s for later use",
                str(compile_range),
            )
        logger.debug(
            "Store the %s-th graph for compile range%s from %s via handle %s",
            graph_index,
            str(compile_range),
            self.compiler.name,
            handle,
        )

    # after compiling the last graph, record the end time
    if graph_index == num_graphs - 1:
        elapsed = time.perf_counter() - compilation_start_time
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
