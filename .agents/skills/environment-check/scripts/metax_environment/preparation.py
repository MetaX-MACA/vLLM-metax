# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""One user authorization, followed by machine verification of the prepared environment.

No target framework imports, installation or Git mutation happens in this module.
"""

import ast
from datetime import datetime, UTC
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[5]


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def now():
    return datetime.now(UTC).isoformat()


def normalize(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def validate_plan(plan):
    if plan.get("schema_version") != 1:
        raise ValueError("Preparation plan schema_version must be 1")
    kernel_variables(plan)
    target = plan["target"]
    if not target["ref"] or target["ref"].startswith("-"):
        raise ValueError("An explicit upstream branch or tag is required")
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", target["commit"]):
        raise ValueError("Resolve the target ref to a full commit before approval")
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?", target["version"]):
        raise ValueError(
            "Specify the target public base version, without a local suffix"
        )
    if target["history"] != "previous-release-range" or not target["previous_release"]:
        raise ValueError("Record the previous release and its history review interval")
    python = plan["python"]
    if python["selection"] not in {"detected_uv", "provided_uv", "create_uv", "global"}:
        raise ValueError("Record the user's Python environment selection")
    for key in ("executable", "prefix", "base_prefix", "base_executable"):
        if not Path(python[key]).is_absolute():
            raise ValueError("Python paths must be absolute: " + key)
    if not python["version"]:
        raise ValueError("Record the approved Python ABI/version")
    if python["selection"] == "create_uv" and not python["system_site_packages"]:
        raise ValueError("The approved new uv environment must inherit image packages")
    sources = plan["sources"]
    for key in ("vllm", "metax", "runtime_cwd"):
        if not Path(sources[key]).is_absolute():
            raise ValueError("Checkout and runtime paths must be absolute")
    if Path(sources["vllm"]).resolve() == Path(sources["metax"]).resolve():
        raise ValueError("Upstream and plugin need separate checkouts")
    if Path(sources["metax"]).resolve() != PROJECT_ROOT:
        raise ValueError("MetaX must be the current project owning this skill")
    if sources["metax_branch"] != "master":
        raise ValueError("MetaX preparation uses the current project's master branch")
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", sources["metax_commit"]):
        raise ValueError("Freeze the MetaX preparation commit")
    if plan["environment_choice"] not in {"reuse", "reinstall"}:
        raise ValueError("Record the framework package reuse/reinstall decision")
    if set(plan["installs"]) != {"vllm", "metax"} or any(
        type(v) is not bool for v in plan["installs"].values()
    ):
        raise ValueError("Record the conditional install decisions for both frameworks")
    if plan["environment_choice"] == "reinstall" and not all(plan["installs"].values()):
        raise ValueError("Declining reuse requires reinstalling both frameworks")
    if plan["git_policy"] != "preserve-worktree":
        raise ValueError("Git preparation must preserve existing work")
    if plan["ordinary_dependencies"] != "allow-with-private-locks":
        raise ValueError("Ordinary dependencies require image-package constraints")
    for key in ("sdk", "gpu", "system"):
        if key not in plan["host_identity"]:
            raise ValueError("Missing approved host identity: " + key)
    private = plan["private_packages"]
    names = [row["normalized_name"] for row in private]
    if len(names) != len(set(names)) or not {"torch", "triton"} <= set(names):
        raise ValueError(
            "Lock unambiguous image Torch, Triton and all private packages"
        )
    if plan["kernel_provider"] == "mcoplib" and "mcoplib" not in names:
        raise ValueError(
            "The mcoplib provider requires the locked image mcoplib installation"
        )
    if plan["kernel_provider"] == "source" and not plan["installs"]["metax"]:
        raise ValueError(
            "Source kernels require an explicit MetaX rebuild with USE_PRECOMPILED_KERNEL=0"
        )
    for row in private:
        if (
            normalize(row["normalized_name"]) != row["normalized_name"]
            or not row["version"]
        ):
            raise ValueError("Invalid private package lock")
    if not isinstance(plan["private_native_identity"], list):
        raise ValueError("Record native identities of locked image distributions")
    if not isinstance(plan["known_issues"], list) or not plan["validation_scope"]:
        raise ValueError("Disclose known issues and the current-chip validation scope")
    return plan


def kernel_variables(plan):
    provider = plan.get("kernel_provider")
    if provider not in {"mcoplib", "source"}:
        raise ValueError(
            "Explicitly confirm kernel_provider: mcoplib or source; do not infer it from package presence"
        )
    return {"USE_PRECOMPILED_KERNEL": "1" if provider == "mcoplib" else "0"}


def backend_issue(plan, origins):
    backend = origins.get("kernel_backend", {})
    if not isinstance(backend, dict):
        return "Child kernel_backend must be an observed JSON object"
    if (
        backend.get("provider") != plan["kernel_provider"]
        or backend.get("use_precompiled_kernel")
        != kernel_variables(plan)["USE_PRECOMPILED_KERNEL"]
    ):
        return "Child kernel provider/USE_PRECOMPILED_KERNEL differs from the approved plan"
    modules = backend.get("loaded_mcoplib_modules")
    if not isinstance(modules, list) or any(
        not isinstance(name, str) for name in modules
    ):
        return "Child must report its actually loaded mcoplib modules"
    if plan["kernel_provider"] == "source" and modules:
        return (
            "Source-kernel execution imported mcoplib; independence is not established"
        )
    return None


def private_locks(report):
    """Include private requirements without a +metax suffix as well as private builds."""
    required = {
        row["normalized_name"]
        for row in report["details"]["requirements"]["entries"]
        if row.get("normalized_name")
    } | {"torch", "triton"}
    return [
        row
        for row in report["identity"]["packages"]
        if row["normalized_name"] in required
        or re.search(r"\+(?:metax|maca)", row["version"], re.I)
    ]


def put_plan(run_dir, plan):
    validate_plan(plan)
    run_dir = Path(run_dir)
    if (run_dir / "PREPARATION.json").exists():
        raise ValueError(
            "Preparation already exists; preserve its authorization and history"
        )
    name = "preparation/plan-" + digest(plan) + ".json"
    write_json(run_dir / name, plan)
    record = {"plan": name, "plan_digest": digest(plan), "status": "awaiting_approval"}
    write_json(run_dir / "PREPARATION.json", record)
    (run_dir / "PREPARATION.md").write_text(
        "# Environment preparation plan\n\nPlan digest: `"
        + digest(plan)
        + "`\n\nApprove this entire plan once, including known issues and conditional "
        "reinstalls. Installation must succeed before adaptation.\n\n```json\n"
        + json.dumps(plan, indent=2, ensure_ascii=False)
        + "\n```\n"
    )
    return record


def load_plan(run_dir, approved=True):
    run_dir = Path(run_dir)
    record = json.loads((run_dir / "PREPARATION.json").read_text())
    plan = validate_plan(json.loads((run_dir / record["plan"]).read_text()))
    if digest(plan) != record["plan_digest"]:
        raise ValueError("Approved preparation plan changed")
    if approved:
        approval = json.loads((run_dir / record["approval"]).read_text())
        if digest(approval) != record["approval_digest"]:
            raise ValueError("Preparation approval artifact changed")
        if approval["plan_digest"] != digest(plan) or not approval["reply"].strip():
            raise ValueError(
                "The actual user reply must approve this exact preparation plan"
            )
    return record, plan


def approve(run_dir, plan_digest, reply, proposal=None):
    record, plan = load_plan(run_dir, approved=False)
    if record.get("approval"):
        raise ValueError(
            "This plan already has an approval; do not replace the user's reply"
        )
    if plan_digest != digest(plan) or not reply.strip():
        raise ValueError(
            "Supply the presented plan digest and the actual explicit user reply"
        )
    approval = {"at": now(), "plan_digest": plan_digest, "reply": reply.strip()}
    if proposal is not None:
        if not proposal.strip():
            raise ValueError("The displayed preparation proposal must not be empty")
        approval["displayed_proposal"] = proposal
    name = "preparation/approval-" + digest(approval) + ".json"
    write_json(Path(run_dir) / name, approval)
    record.update(approval=name, approval_digest=digest(approval), status="preparing")
    write_json(Path(run_dir) / "PREPARATION.json", record)


def git(root, *args):
    return subprocess.check_output(
        ["git", "-C", str(root), *args], text=True, stderr=subprocess.PIPE
    ).strip()


def fixed_version(source):
    tree = ast.parse((Path(source) / "setup.py").read_text())
    returns = [
        node
        for function in tree.body
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
        and function.name == "fixed_version_scheme"
        for node in ast.walk(function)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ]
    if len(returns) != 1:
        raise ValueError(
            "Review setup.py: expected one fixed_version_scheme string return"
        )
    return returns[0].value


def check_sources(plan, preparation=False):
    sources, target = plan["sources"], plan["target"]
    if git(sources["vllm"], "rev-parse", "HEAD") != target["commit"]:
        raise ValueError("Upstream checkout does not match the approved target commit")
    if (
        git(sources["vllm"], "rev-parse", target["ref"] + "^{commit}")
        != target["commit"]
    ):
        raise ValueError("Upstream ref no longer resolves to the approved target")
    if git(sources["vllm"], "status", "--porcelain", "--untracked-files=normal"):
        raise ValueError(
            "Target vLLM checkout has unreviewed local changes; preserve it and use a clean checkout"
        )
    if git(sources["metax"], "branch", "--show-current") != sources["metax_branch"]:
        raise ValueError("MetaX must be on the approved master branch")
    if (
        preparation
        and git(sources["metax"], "rev-parse", "HEAD") != sources["metax_commit"]
    ):
        raise ValueError("MetaX preparation commit changed")


def verify_identity(plan, report):
    """Check Python, host and image-package locks before and after installation."""
    identity = report["identity"]
    for key in ("executable", "prefix", "base_prefix", "version"):
        if identity["python"].get(key) != plan["python"][key]:
            raise ValueError("Selected Python identity differs from the plan: " + key)
    for key, expected in plan["host_identity"].items():
        if identity.get(key) != expected:
            raise ValueError(
                "Host identity changed outside the preparation plan: " + key
            )
    expected = plan["private_packages"]
    names = {row["normalized_name"] for row in expected}
    actual = [row for row in identity["packages"] if row["normalized_name"] in names]
    if sorted(actual, key=digest) != sorted(expected, key=digest):
        raise ValueError("Locked image package identity changed or is ambiguous")
    missing_locks = {row["normalized_name"] for row in private_locks(report)} - names
    if missing_locks:
        raise ValueError(
            "Unapproved private packages: " + ", ".join(sorted(missing_locks))
        )
    native = [
        row
        for row in identity["native_files"]["distributions"]
        if row["normalized_name"] in names
    ]
    if sorted(native, key=digest) != sorted(
        plan["private_native_identity"], key=digest
    ):
        raise ValueError("Locked image native files changed")


def installed_candidates(package, source):
    """Ignore source-only egg-info; it is not a second installed distribution."""
    result = []
    for row in package["installed_distributions"]:
        locations = row.get("metadata_paths", [])
        if locations and all(
            path.endswith(".egg-info")
            and Path(path).resolve().is_relative_to(Path(source).resolve())
            for path in locations
        ):
            continue
        result.append(row)
    return result


def verify_report(plan, report):
    """Check installation predicates; this is not model compatibility evidence."""
    verify_identity(plan, report)
    if (
        report["identity"].get("runtime_environment", {}).get("USE_PRECOMPILED_KERNEL")
        != kernel_variables(plan)["USE_PRECOMPILED_KERNEL"]
    ):
        raise ValueError(
            "Runtime USE_PRECOMPILED_KERNEL differs from the approved kernel provider"
        )
    sources = plan["sources"]
    check_sources(plan)
    common = Path(sources["metax"]) / "requirements/common.txt"
    if (
        common.read_bytes()
        != (Path(sources["vllm"]) / "requirements/common.txt").read_bytes()
    ):
        raise ValueError("MetaX common.txt must exactly match the target common.txt")
    if fixed_version(sources["metax"]).value != plan["target"]["version"]:
        raise ValueError("MetaX fixed base version differs from the target")
    probe = report["details"]["source_probe"]
    for module, source_key in (("vllm", "vllm"), ("vllm_metax", "metax")):
        package = probe[module]
        distributions = installed_candidates(package, sources[source_key])
        if len(distributions) != 1:
            raise ValueError(
                "Framework installation is missing or ambiguous: " + module
            )
        dist = distributions[0]
        if dist["version"].split("+", 1)[0] != plan["target"]["version"]:
            raise ValueError(
                "Installed framework version differs from the target: " + module
            )
        provenance = dist.get("provenance") or {}
        source = str(Path(sources[source_key]).resolve())
        if provenance.get("local_source_or_archive") != source:
            raise ValueError(
                "Framework distribution was not installed from the selected source: "
                + module
            )
        if module == "vllm_metax":
            if (
                not provenance.get("editable")
                or not package["effective_import_probe"]["includes_local_package"]
            ):
                raise ValueError(
                    "MetaX must be editable and resolve to the current project"
                )
        elif not package["effective_import_probe"]["includes_local_package"]:
            trees = [
                tree
                for tree in dist["package_trees"]
                if tree["selected_by_import_probe"]
            ]
            if not trees:
                raise ValueError(
                    "Cannot establish actual installed vLLM/source correspondence"
                )
            for tree in trees:
                comparison = tree["comparison_to_local"]
                changes = (
                    set(comparison.get("changed_files", []))
                    | set(comparison.get("local_only_files", []))
                    | set(comparison.get("installed_only_files", []))
                )
                if comparison.get("status") not in {
                    "identical",
                    "different",
                } or changes - {"_version.py", "version.py"}:
                    raise ValueError(
                        "Installed vLLM differs substantively from the target source"
                    )


def verify(run_dir, report):
    run_dir = Path(run_dir)
    record, plan = load_plan(run_dir)
    report_name = "preparation/report-" + digest(report) + ".json"
    write_json(run_dir / report_name, report)
    record["last_report"] = report_name
    try:
        for package, required in plan["installs"].items():
            if required:
                receipt = json.loads(
                    (
                        run_dir / "preparation" / ("install-" + package + ".json")
                    ).read_text()
                )
                if (
                    receipt.get("status") != "installed"
                    or receipt.get("plan_digest") != digest(plan)
                    or receipt.get("kernel_provider") != plan["kernel_provider"]
                    or receipt.get("kernel_environment") != kernel_variables(plan)
                ):
                    raise ValueError(
                        "Required source installation did not succeed: " + package
                    )
        journal = run_dir / "preparation/temporary-requirements.json"
        if journal.exists() and json.loads(journal.read_text())["status"] != "restored":
            raise ValueError(
                "Temporary dependency edits must be recovered before verification"
            )
        verify_report(plan, report)
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as exc:
        record.update(status="blocked", blocker=str(exc), checked_at=now())
        write_json(run_dir / "PREPARATION.json", record)
        raise
    verified = {
        "at": now(),
        "plan_digest": digest(plan),
        "approval_digest": record["approval_digest"],
        "environment_fingerprint": digest(report["identity"]),
        "report": report,
    }
    name = "preparation/verified-" + digest(verified) + ".json"
    write_json(run_dir / name, verified)
    record.pop("blocker", None)
    record.update(
        status="ready",
        ever_ready=True,
        verification=name,
        verification_digest=digest(verified),
    )
    write_json(run_dir / "PREPARATION.json", record)
    return record


def require_ready(run_dir, inputs, fingerprint):
    """Used by the upgrade state tool, including before any adaptation edits."""
    try:
        record, plan = load_plan(run_dir)
        if record["status"] != "ready":
            raise ValueError("Installation/preparation has not completed successfully")
        verified = json.loads((Path(run_dir) / record["verification"]).read_text())
        if digest(verified) != record["verification_digest"]:
            raise ValueError("Environment verification artifact changed")
        if (
            verified["plan_digest"] != digest(plan)
            or verified["approval_digest"] != record["approval_digest"]
        ):
            raise ValueError(
                "Environment verification belongs to another preparation plan"
            )
        if verified["environment_fingerprint"] != fingerprint:
            raise ValueError(
                "Prepared environment drifted; verify against the approved plan without asking again"
            )
        for key, source_key in (
            ("vllm_source", "vllm"),
            ("metax_source", "metax"),
            ("runtime_cwd", "runtime_cwd"),
        ):
            if str(Path(inputs[key]).resolve()) != str(
                Path(plan["sources"][source_key]).resolve()
            ):
                raise ValueError(
                    "Upgrade inputs differ from the approved preparation paths"
                )
        if sys.executable != plan["python"]["executable"]:
            raise ValueError("Run the upgrade tool with the approved interpreter")
        return {
            "mode": "preparation-plan",
            "fingerprint": fingerprint,
            "plan_digest": digest(plan),
            "approval_record": record["approval"],
            "verification_record": record["verification"],
        }
    except (KeyError, OSError) as exc:
        raise ValueError(
            "An approved, successfully verified environment-check plan is required"
        ) from exc
