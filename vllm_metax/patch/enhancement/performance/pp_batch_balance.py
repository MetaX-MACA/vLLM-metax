# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
#
# -----------------------------------------------------------------------------
# Note: Automatically balance synchronous MRv2 PP decode batches. The dynamic
#     target is ceil(min(global request capacity, unfinished requests) / PP).
#     Preserve prefill budgets and in-flight bookkeeping; permit overdue decodes
#     to exceed the target rather than starving them. Other modes delegate.
#
# Affected versions: vLLM 0.33.0 Scheduler.schedule RUNNING/WAITING allocation
#     paths. No installed vLLM source files are modified.
#
# Remove at: Native PP balancing passes the MetaX correctness/throughput matrix
#     at concurrency 16, 32 and 64, including arrivals and request draining.
# -----------------------------------------------------------------------------

import ast
import inspect
import textwrap
from functools import wraps
from typing import Any

from vllm.logger import init_logger
from vllm.v1.core.sched.scheduler import Scheduler

from vllm_metax.patch.utils import PATCH_MARKER, patch

logger = init_logger(__name__)


def _is_decode(request, num_tokens, num_computed_tokens):
    # A final one-token prefill chunk and multi-token catch-up remain prefill.
    return num_tokens == 1 and num_computed_tokens >= request.num_prompt_tokens


class _PPBatchBalance:
    def __init__(self, pp_size):
        self.pp_size = pp_size
        self.deferred = {}

    def begin(self, scheduler):
        # Include in-flight requests: counting only currently ready requests
        # would miss the other PP batch and underestimate the target.
        active = min(
            scheduler.max_num_running_reqs,
            scheduler.get_num_unfinished_requests(),
        )
        self.target = max(1, (active + self.pp_size - 1) // self.pp_size)
        self.deferred = {
            rid: count
            for rid, count in self.deferred.items()
            if rid in scheduler.requests
        }
        self.decodes = set()
        self.skipped = set()

    def defer(self, request, num_tokens, num_computed_tokens, scheduled):
        if not _is_decode(request, num_tokens, num_computed_tokens):
            return False
        # A row preempted and removed by upstream no longer consumes a slot.
        if len(self.decodes.intersection(scheduled)) < self.target:
            return False
        rid = request.request_id
        if self.deferred.get(rid, 0) >= self.pp_size:
            # A soft target: ready requests cannot be deferred indefinitely.
            return False
        self.skipped.add(rid)
        return True

    def record(self, request, num_tokens, num_computed_tokens):
        if _is_decode(request, num_tokens, num_computed_tokens):
            self.decodes.add(request.request_id)

    def finish(self, scheduled):
        for rid in self.skipped.difference(scheduled):
            self.deferred[rid] = self.deferred.get(rid, 0) + 1
        for rid in scheduled:
            self.deferred.pop(rid, None)


def _transform_schedule(tree):
    """Add two before-allocation guards and two after-insertion counters."""
    if (
        len(tree.body) != 1
        or not isinstance(tree.body[0], ast.FunctionDef)
        or tree.body[0].decorator_list
    ):
        raise RuntimeError("PP balancing requires an undecorated Scheduler.schedule")
    loops = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.While):
            continue
        names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
        attrs = {n.attr for n in ast.walk(node.test) if isinstance(n, ast.Attribute)}
        loop_attrs = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
        calls = {
            n.func.id
            for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        kind = None
        if {"req_index", "token_budget"} <= names and "running" in attrs:
            kind = "running"
        if (
            "token_budget" in names
            and "peek_request" in loop_attrs
            and "skip_request" in calls
        ):
            kind = "waiting"
        if kind is not None:
            if kind in loops:
                raise RuntimeError("PP balancing found ambiguous scheduling loops")
            loops[kind] = node
    if set(loops) != {"running", "waiting"}:
        raise RuntimeError("PP balancing requires vLLM 0.33's two scheduling loops")
    edits = []
    for kind, loop in loops.items():
        allocations = [
            (i, node)
            for i, node in enumerate(loop.body)
            if any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr == "allocate_slots"
                for n in ast.walk(node)
            )
        ]
        insertions = [
            (i, node)
            for i, node in enumerate(loop.body)
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and ast.unparse(node.targets[0]) == "num_scheduled_tokens[request_id]"
            and ast.unparse(node.value) == "num_new_tokens"
        ]
        if len(allocations) != 1 or len(insertions) != 1:
            raise RuntimeError("PP balancing allocation/insertion source shape changed")
        allocation_index, allocation = allocations[0]
        insertion_index, insertion = insertions[0]
        if insertion_index <= allocation_index:
            raise RuntimeError(
                "PP balancing requires allocation before output insertion"
            )
        computed = (
            "request.num_computed_tokens"
            if kind == "running"
            else "num_computed_tokens"
        )
        skip = (
            "req_index += 1\n    continue"
            if kind == "running"
            else "skip_request(request_queue)\n    continue"
        )
        guard = ast.parse(
            "if self._metax_pp_balance.defer(\n"
            f"    request, num_new_tokens, {computed}, num_scheduled_tokens):\n"
            f"    {skip}\n"
        ).body[0]
        record = ast.parse(
            f"self._metax_pp_balance.record(request, num_new_tokens, {computed})"
        ).body[0]
        for new, anchor in ((guard, allocation), (record, insertion)):
            for child in ast.walk(new):
                ast.copy_location(child, anchor)
        edits.append((loop, allocation_index, insertion_index, guard, record))
    # /-------------------- MetaX Modification --------------------\
    for loop, allocation_index, insertion_index, guard, record in edits:
        loop.body.insert(insertion_index + 1, record)
        loop.body.insert(allocation_index, guard)
    # \-------------------- MetaX Modification --------------------/
    return ast.fix_missing_locations(tree)


def _schedule_with_balance(original):
    if original.__closure__:
        raise RuntimeError("PP balancing does not support a scheduler closure")
    lines, first_line = inspect.getsourcelines(original)
    source = textwrap.dedent("".join(lines))
    if any(
        marker in source
        for marker in (
            "_codex_pp_batch_cap",
            "_metax_pp_batch_max_reqs",
            "_metax_pp_balance",
            "max_num_scheduled_decodes",
            "_get_max_num_scheduled_reqs",
        )
    ):
        raise RuntimeError("Restore/remove the previous PP scheduler patch first")
    tree = _transform_schedule(ast.parse(source))
    ast.increment_lineno(tree, first_line - 1)
    namespace: dict[str, Any] = {}
    exec(
        compile(tree, inspect.getfile(original), "exec"),
        original.__globals__,
        namespace,
    )
    return namespace[original.__name__]


def _supports_balancing(config):
    return (
        config.parallel_config.pipeline_parallel_size > 1
        and config.parallel_config.data_parallel_size == 1
        and config.scheduler_config.async_scheduling is False
        and config.scheduler_config.policy == "fcfs"
        and config.speculative_config is None
        and config.use_v2_model_runner
    )


_original_schedule = Scheduler.schedule
if getattr(_original_schedule, PATCH_MARKER, None):
    raise RuntimeError(
        "PP balancing conflicts with an existing Scheduler.schedule patch"
    )
_balanced_schedule: Any = None


@patch("vllm.v1.core.sched.scheduler", "Scheduler.schedule")
@wraps(_original_schedule)
def _schedule_with_pp_balance(self, *args, **kwargs):
    global _balanced_schedule
    state = getattr(self, "_metax_pp_balance", None)
    if state is None:
        config = self.vllm_config
        if not _supports_balancing(config):
            return _original_schedule(self, *args, **kwargs)
        # Compile once, only when a supported PP scheduler actually executes.
        if _balanced_schedule is None:
            _balanced_schedule = _schedule_with_balance(_original_schedule)
        state = _PPBatchBalance(config.parallel_config.pipeline_parallel_size)
        self._metax_pp_balance = state
        logger.info(
            "PP decode batch balancing enabled automatically: pp=%d "
            "global_running_limit=%d",
            state.pp_size,
            self.max_num_running_reqs,
        )
    state.begin(self)
    output = _balanced_schedule(self, *args, **kwargs)
    state.finish(output.num_scheduled_tokens)
    return output
