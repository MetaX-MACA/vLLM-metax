# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Approved source preparation and installation with recoverable temporary edits."""

import base64
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import subprocess

from . import preparation as prep


@contextmanager
def mutation_lock(run_dir):
    path = Path(run_dir) / "preparation/mutation.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def restore(journal):
    journal = Path(journal)
    saved = json.loads(journal.read_text())
    if saved["status"] == "restored":
        return
    root = Path(saved["source"])
    for name, record in saved["files"].items():
        path = root / name
        if path.is_symlink():
            raise ValueError("Refuse to restore through a changed symlink: " + name)
        path.write_bytes(base64.b64decode(record["bytes"]))
        path.chmod(record["mode"])
    saved["status"] = "restored"
    prep.write_json(journal, saved)


@contextmanager
def temporary_requirements(source, journal):
    """Restore exact original bytes/modes, including pre-existing dirty contents."""
    source, journal = Path(source), Path(journal)
    if journal.exists() and json.loads(journal.read_text())["status"] != "restored":
        raise ValueError("An interrupted dependency cleanup requires recovery first")
    files = [
        *sorted((source / "requirements").glob("*.txt")),
        source / "pyproject.toml",
    ]
    saved = {"source": str(source.resolve()), "status": "temporary", "files": {}}
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise ValueError(
                "Dependency suppression needs ordinary existing files: " + str(path)
            )
        saved["files"][str(path.relative_to(source))] = {
            "bytes": base64.b64encode(path.read_bytes()).decode(),
            "mode": stat.S_IMODE(path.stat().st_mode),
        }
    prep.write_json(journal, saved)
    try:
        yield
    finally:
        restore(journal)


def pip_command(plan, operation, *args):
    python = plan["python"]["executable"]
    probe = subprocess.run(
        [
            python,
            "-c",
            "import importlib.util; raise SystemExit(importlib.util.find_spec('pip') is None)",
        ],
        capture_output=True,
    )
    if probe.returncode == 0:
        return [python, "-m", "pip", operation, *args]
    uv = shutil.which("uv")
    if not uv:
        raise ValueError("The approved interpreter has no pip and uv is unavailable")
    return [uv, "pip", operation, "--python", python, *args]


def constraints(run_dir, plan):
    path = Path(run_dir) / "preparation/private-constraints.txt"
    path.write_text(
        "".join(
            row["normalized_name"] + "==" + row["version"] + "\n"
            for row in sorted(
                plan["private_packages"], key=lambda row: row["normalized_name"]
            )
        )
    )
    return path


def sync_sources(run_dir):
    _, plan = prep.load_plan(run_dir)
    prep.check_sources(plan, preparation=True)
    source = Path(plan["sources"]["metax"])
    # Preserve all SCM/local suffix logic and formatting outside the base literal.
    path = source / "setup.py"
    text = path.read_text()
    literal = prep.fixed_version(source)
    lines = text.splitlines(keepends=True)
    start = (
        sum(len(line.encode()) for line in lines[: literal.lineno - 1])
        + literal.col_offset
    )
    end = (
        sum(len(line.encode()) for line in lines[: literal.end_lineno - 1])
        + literal.end_col_offset
    )
    encoded = text.encode()
    quote = encoded[start : start + 1].decode()
    replacement = (quote + plan["target"]["version"] + quote).encode()
    path.write_bytes(encoded[:start] + replacement + encoded[end:])
    (source / "requirements/common.txt").write_bytes(
        (Path(plan["sources"]["vllm"]) / "requirements/common.txt").read_bytes()
    )


def install(run_dir, package):
    run_dir = Path(run_dir)
    record, plan = prep.load_plan(run_dir)
    if not plan["installs"][package]:
        raise ValueError(
            "Installation of this framework package is outside the approved plan"
        )
    record.update(status="preparing")
    prep.write_json(run_dir / "PREPARATION.json", record)
    prep.check_sources(plan, preparation=not record.get("ever_ready", False))
    if sys.executable != plan["python"]["executable"]:
        raise ValueError("Use the approved interpreter for installation")
    sys.path.insert(
        0, str(Path(__file__).resolve().parents[3] / "common-sense/scripts")
    )
    from metax_upgrade.environment import collect_environment

    sources = plan["sources"]
    report = collect_environment(
        Path(sources["vllm"]),
        Path(sources["metax"]),
        Path(sources["runtime_cwd"]),
        include_collect_env=False,
    )
    prep.verify_identity(plan, report)
    source = Path(plan["sources"][package])
    folder = run_dir / "preparation"
    lockfile = constraints(run_dir, plan)
    env = os.environ.copy()
    env["VLLM_PLUGINS"] = ""
    env.update(prep.kernel_variables(plan))
    env.pop("VLLM_TARGET_DEVICE", None)
    env["PIP_CONSTRAINT"] = str(lockfile)
    env["UV_CONSTRAINT"] = str(lockfile)
    receipt = {
        "package": package,
        "kernel_provider": plan["kernel_provider"],
        "kernel_environment": prep.kernel_variables(plan),
        "plan_digest": record["plan_digest"],
        "at": prep.now(),
        "status": "running",
        "commands": [],
    }
    receipt_path = folder / ("install-" + package + ".json")
    receipt["log"] = (
        "preparation/install-" + package + "-" + prep.digest(receipt)[:16] + ".log"
    )
    prep.write_json(receipt_path, receipt)
    prep.write_json(
        folder / "install-history" / (prep.digest(receipt) + ".json"), receipt
    )

    def run(command, *, metax=False):
        if metax:
            command = [
                "bash",
                "-c",
                'source "$1" && shift && exec "$@"',
                "bash",
                str(source / "env.sh"),
                *command,
            ]
        receipt["commands"].append(command)
        prep.write_json(receipt_path, receipt)
        with (run_dir / receipt["log"]).open("a") as log:
            subprocess.run(
                command,
                cwd=source,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    try:
        if plan["environment_choice"] == "reinstall":
            module = "vllm" if package == "vllm" else "vllm_metax"
            for dist in prep.installed_candidates(
                report["details"]["source_probe"][module], source
            ):
                if (
                    not Path(dist["installation_root"])
                    .resolve()
                    .is_relative_to(Path(plan["python"]["prefix"]).resolve())
                ):
                    raise ValueError(
                        "Refuse to uninstall an inherited framework package from the image environment"
                    )
            command = pip_command(
                plan, "uninstall", "vllm" if package == "vllm" else "vllm-metax"
            )
            if "-m" in command:
                command.append("-y")
            run(command)
        if package == "vllm":
            env["VLLM_TARGET_DEVICE"] = "empty"
            run(
                pip_command(
                    plan, "install", "-v", ".", "--no-build-isolation", "--no-deps"
                )
            )
        else:
            with temporary_requirements(source, folder / "temporary-requirements.json"):
                run([plan["python"]["executable"], "use_existing_metax.py"], metax=True)
                run(
                    pip_command(
                        plan,
                        "install",
                        "-v",
                        "-e",
                        ".",
                        "--no-build-isolation",
                        "--constraint",
                        str(lockfile),
                    ),
                    metax=True,
                )
    except BaseException as exc:
        receipt.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
        raise
    else:
        receipt["status"] = "installed"
    finally:
        receipt["finished_at"] = prep.now()
        log_path = run_dir / receipt["log"]
        if log_path.exists():
            receipt["log_sha256"] = hashlib.sha256(log_path.read_bytes()).hexdigest()
        prep.write_json(receipt_path, receipt)
        prep.write_json(
            folder / "install-history" / (prep.digest(receipt) + ".json"), receipt
        )
        record.update(
            status="preparing" if receipt["status"] == "installed" else "blocked"
        )
        prep.write_json(run_dir / "PREPARATION.json", record)
