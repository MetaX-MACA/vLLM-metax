# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Source and runtime obligations for replacements of upstream APIs."""

SURFACES = {"python": {"signature"}, "custom_op": {"schema", "fake_signature"}}
REPLACEMENT_PATHS = ("vllm_metax/patch/", "vllm_metax/registry/")


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _identity(record):
    return record["local"], record["local_symbol"]


def _check_surface(value, surface):
    return (
        isinstance(value, dict)
        and value.keys() == SURFACES[surface]
        and all(
            _text(item) or (field == "fake_signature" and item is None)
            for field, item in value.items()
        )
    )


def definitions(task, document):
    contracts = document.get("api_contracts", [])
    no_replacements = document.get("no_api_replacements", {})
    if not isinstance(contracts, list) or not isinstance(no_replacements, dict):
        raise ValueError(
            "API contracts need a list and no_api_replacements needs an object"
        )
    if any(
        path not in task["files"] or not _text(reason)
        for path, reason in no_replacements.items()
    ):
        raise ValueError(
            "Each no_api_replacements entry needs an owned file and review reason"
        )
    ids = set()
    for row in contracts:
        if not isinstance(row, dict) or not all(
            _text(row.get(field))
            for field in ("id", "local", "local_symbol", "surface", "mode")
        ):
            raise ValueError(
                "API contracts need an ID, local file/symbol, surface and mode"
            )
        binding = row.get("binding")
        if (
            not isinstance(binding, dict)
            or binding.keys() != {"entry", "callable"}
            or not all(_text(value) for value in binding.values())
        ):
            raise ValueError(
                "API contracts need the bound attribute or registry key and effective callable"
            )
        if row["local"] not in task["files"] or row["local"] in no_replacements:
            raise ValueError("API contract needs an owned file with a replacement")
        if row["id"] in ids:
            raise ValueError("Duplicate API contract ID")
        ids.add(row["id"])
        if row["surface"] not in SURFACES or not _check_surface(
            row.get("target"), row["surface"]
        ):
            raise ValueError(
                "Freeze the target Python signature or the CustomOp schema and fake signature"
            )
        if row["mode"] == "adapter":
            if not all(
                _text(row.get(field)) for field in ("reason", "removal_condition")
            ):
                raise ValueError(
                    "API adapter needs an incompatibility reason and removal condition"
                )
        elif row["mode"] != "exact":
            raise ValueError("API contract mode must be exact or adapter")
        required_cases = {"valid_call", "invalid_call"}
        if row["mode"] == "adapter":
            required_cases.add("adapter_forwarding")
        cases = row.get("cases")
        if (
            not isinstance(cases, list)
            or not all(_text(case) for case in cases)
            or len(set(cases)) != len(cases)
            or not required_cases <= set(cases)
        ):
            raise ValueError(
                "API cases need unique IDs including "
                + ", ".join(sorted(required_cases))
            )
    return contracts


def source_issues(task):
    checks = task.get("local_checks", {})
    try:
        contracts = definitions(task, checks)
    except ValueError as exc:
        return [str(exc)]
    changed = {"api", "dispatch", "lifecycle"} & set(checks.get("changes", []))
    covered_files = {row["local"] for row in contracts} | checks.get(
        "no_api_replacements", {}
    ).keys()
    applicable = {
        path
        for path in task["files"]
        if path.endswith(".py")
        and (
            path.startswith(REPLACEMENT_PATHS)
            or (changed and path.startswith("vllm_metax/"))
        )
    }
    issues = [
        "Inventory replacement APIs with checks-put for: " + path
        for path in sorted(applicable - covered_files)
    ]
    mappings = {
        (row["local"], row.get("local_symbol"))
        for row in task.get("upstream_mapping", {}).get("mappings", [])
        if row.get("local_symbol") and not row.get("upstream", {}).get("absent")
    }
    declared = {_identity(row) for row in contracts}
    issues.extend(
        f"API contract needs a pinned upstream mapping for {path}:{symbol}"
        for path, symbol in sorted(declared - mappings)
    )
    issues.extend(
        f"Mapped replacement needs an API contract for {path}:{symbol}"
        for path, symbol in sorted(mappings - declared)
        if path in applicable
    )
    if contracts and "contract" not in task["requires"]:
        issues.append("Replacement APIs require component contract evidence")
    return issues


def report_issues(contracts, report):
    observed = report.get("api_contracts") if isinstance(report, dict) else None
    if not isinstance(observed, dict) or observed.keys() != {
        row["id"] for row in contracts
    }:
        return ["Contract check must report exactly the declared replacement APIs"]
    issues = []
    for row in contracts:
        key = row["id"]
        result = observed[key]
        if not isinstance(result, dict):
            issues.append(f"API contract {key} needs structured runtime observations")
            continue
        if result.get("binding") != row["binding"]:
            issues.append(
                f"API contract {key} selected a different callable or registry entry"
            )
        if result.get("upstream") != row["target"]:
            issues.append(
                f"API contract {key} observed a different target upstream surface"
            )
        active = result.get("active")
        must_match = row["mode"] == "exact" or row["surface"] == "custom_op"
        if not _check_surface(active, row["surface"]) or (
            must_match and active != row["target"]
        ):
            issues.append(
                f"API contract {key} active signature or schema is incompatible"
            )
        cases = result.get("cases")
        if (
            not isinstance(cases, dict)
            or cases.keys() != set(row["cases"])
            or any(passed is not True for passed in cases.values())
        ):
            issues.append(
                f"API contract {key} needs passing evidence for every declared caller case"
            )
    return issues
