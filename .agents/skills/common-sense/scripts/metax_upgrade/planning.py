# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Recorded dependency analysis and change-specific component obligations."""

NUMERICAL = {"layout", "quantization", "indexing", "numerics", "communication"}
CHANGES = NUMERICAL | {"api", "dispatch", "lifecycle", "none"}


def _store():
    if __package__:
        from . import state
    else:
        import state
    return state


def put_analysis(state, document):
    files = document.get("files")
    inventory = state["inventory"]
    if (
        not isinstance(files, list)
        or not all(isinstance(p, str) for p in files)
        or not set(inventory) <= set(files)
    ):
        raise ValueError("Dependency analysis must cover the entire file inventory")
    if not isinstance(document.get("summary"), str) or not document["summary"].strip():
        raise ValueError("Dependency analysis needs a reviewed contract summary")
    models = document.get("models", [])
    if (
        not isinstance(models, list)
        or not all(isinstance(p, str) for p in models)
        or not set(state.get("model_inventory", {})) <= set(models)
    ):
        raise ValueError("Dependency analysis must include every baseline model sample")
    contracts = document.get("model_contracts", {})
    if not isinstance(contracts, dict) or any(
        not isinstance(contracts.get(model), dict)
        or any(
            not isinstance(contracts[model].get(field), str)
            or not contracts[model][field].strip()
            for field in ("framework_interfaces", "cache_layout", "quantization")
        )
        for model in state.get("model_inventory", {})
    ):
        raise ValueError(
            "Review framework interfaces, cache layout and quantization for every baseline model before framework edits"
        )
    edges = document.get("edges")
    if not isinstance(edges, list):
        raise ValueError(
            "Dependency analysis needs edges (an empty graph needs a justification in summary)"
        )
    for edge in edges:
        if (
            not isinstance(edge, dict)
            or edge.get("producer") not in inventory
            or edge.get("consumer") not in inventory
            or not isinstance(edge.get("contract"), str)
            or not edge["contract"].strip()
        ):
            raise ValueError(
                "Each dependency edge needs inventoried producer/consumer files and a contract"
            )
    cycles = document.get("cycles")
    if not isinstance(cycles, list):
        raise ValueError(
            "Dependency analysis must declare cycles and coordinated handling"
        )
    for cycle in cycles:
        if (
            not isinstance(cycle, dict)
            or not cycle.get("files")
            or not set(cycle["files"]) <= set(inventory)
            or not cycle.get("strategy")
        ):
            raise ValueError(
                "Each dependency cycle needs files and a coordinated strategy"
            )
    graph = {path: set() for path in inventory}
    for edge in edges:
        graph[edge["producer"]].add(edge["consumer"])
    reachable = {}
    for start in graph:
        seen, pending = set(), list(graph[start])
        while pending:
            node = pending.pop()
            if node not in seen:
                seen.add(node)
                pending.extend(graph[node] - seen)
        reachable[start] = seen
    for start in graph:
        if start in reachable[start]:
            group = {node for node in reachable[start] if start in reachable[node]}
            if not any(group <= set(cycle["files"]) for cycle in cycles):
                raise ValueError(
                    "Dependency cycle needs a coordinated strategy: "
                    + ", ".join(sorted(group))
                )
    # Retain history; expected source port edits do not invalidate the planning record.
    state.setdefault("analysis_history", []).append(state.get("dependency_analysis"))
    state["dependency_analysis"] = {
        **document,
        "environment": state["environment"]["fingerprint"],
        "model_configs": {
            key: row["config_hash"]
            for key, row in state.get("model_inventory", {}).items()
        },
    }
    state["completed_at"] = None


def analysis_issues(state):
    if state["inputs"]["mode"] != "full":
        return []
    document = state.get("dependency_analysis", {})
    if not document:
        return [
            "Complete global dependency/model-contract analysis with analysis-put before adaptation"
        ]
    if document.get("environment") != state["environment"]["fingerprint"]:
        return [
            "Recheck dependency/model contracts for the changed confirmed environment"
        ]
    if document.get("model_configs", {}) != {
        key: row["config_hash"] for key, row in state.get("model_inventory", {}).items()
    }:
        return [
            "Recheck dependency/model contracts for changed model sample configurations"
        ]
    missing = set(state["inventory"]) - set(document.get("files", []))
    missing_models = set(state.get("model_inventory", {})) - set(
        document.get("models", [])
    )
    return (
        [
            "Update dependency analysis for added files/models: "
            + ", ".join(sorted(missing | missing_models))
        ]
        if missing or missing_models
        else []
    )


def put_checks(state, task_id, document):
    task = state["tasks"][task_id]
    changes = document.get("changes")
    if (
        not isinstance(changes, list)
        or not changes
        or not all(isinstance(v, str) and v in CHANGES for v in changes)
        or ("none" in changes and len(changes) != 1)
    ):
        raise ValueError(
            "Declare semantic change kinds; none cannot accompany another kind"
        )
    if not isinstance(document.get("reason"), str) or not document["reason"].strip():
        raise ValueError("Local check selection needs a reviewed reason")
    required = {"static", "review"}
    if set(changes) - {"none"}:
        required.add("contract")
    if set(changes) & NUMERICAL:
        required.add("component_numeric")
    if required - {"static", "review"} and not document.get("cases"):
        raise ValueError("Semantic changes need discriminating component cases")
    defer = document.get("defer_numeric")
    if defer:
        matches = [
            row
            for row in state.get("workflow", {}).get("unverified", [])
            if row.get("feature") == defer.get("feature")
            and row.get("chip") == defer.get("chip")
            and set(task["files"]) <= set(row.get("files", []))
        ]
        if (
            "component_numeric" not in required
            or not matches
            or not defer.get("reason")
        ):
            raise ValueError(
                "Numerical deferral needs a matching explicit unavailable-chip coverage record"
            )
    store = _store()
    store.invalidate(state, {task_id}, "Component check policy updated")
    task["local_checks"] = {
        **document,
        "required": sorted(required),
        "environment": state["environment"]["fingerprint"],
    }
    task["requires"] = sorted(set(task["requires"]) | required)


def deferred_kinds(state, task_id):
    task = state["tasks"][task_id]
    checks = task.get("local_checks", {})
    defer = checks.get("defer_numeric")
    if not defer or checks.get("environment") != state["environment"]["fingerprint"]:
        return set()
    for row in state.get("workflow", {}).get("unverified", []):
        if (
            row.get("feature") == defer.get("feature")
            and row.get("chip") == defer.get("chip")
            and set(task["files"]) <= set(row.get("files", []))
        ):
            return {"component_numeric"}
    return set()


def source_issues(state, task_id):
    if (
        state["inputs"]["mode"] != "full"
        and state["tasks"][task_id].get("phase") != "models"
    ):
        return []
    task = state["tasks"][task_id]
    if task["files"] and not task.get("local_checks"):
        return ["Classify source changes and required component checks with checks-put"]
    checks = task.get("local_checks", {})
    if checks and not set(checks["required"]) <= set(task["requires"]):
        return ["Required component evidence is missing from task acceptance"]
    return []
