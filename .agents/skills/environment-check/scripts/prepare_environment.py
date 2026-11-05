#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Record one preparation authorization, then install and verify its exact environment."""

import argparse
import json
from pathlib import Path
import sys
import subprocess

from metax_environment import install, preparation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    for action in (
        "plan",
        "approve",
        "sync-sources",
        "install",
        "recover",
        "verify",
        "status",
    ):
        command = commands.add_parser(action)
        command.add_argument("--run-dir", type=Path, required=True)
        if action == "plan":
            command.add_argument("--file", type=Path, required=True)
        elif action == "approve":
            command.add_argument("--plan-digest", required=True)
            command.add_argument("--user-response-file", type=Path, required=True)
            command.add_argument("--proposal-file", type=Path)
        elif action == "install":
            command.add_argument("--package", choices=("vllm", "metax"), required=True)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    with install.mutation_lock(run_dir):
        try:
            if args.action == "plan":
                preparation.put_plan(run_dir, json.loads(args.file.read_text()))
            elif args.action == "approve":
                preparation.approve(
                    run_dir,
                    args.plan_digest,
                    args.user_response_file.read_text(),
                    args.proposal_file.read_text() if args.proposal_file else None,
                )
            elif args.action == "sync-sources":
                install.sync_sources(run_dir)
            elif args.action == "install":
                install.install(run_dir, args.package)
            elif args.action == "recover":
                preparation.load_plan(run_dir)
                install.restore(run_dir / "preparation/temporary-requirements.json")
            elif args.action == "verify":
                _, plan = preparation.load_plan(run_dir)
                if sys.executable != plan["python"]["executable"]:
                    raise ValueError(
                        "Use the approved interpreter for final collection"
                    )
                sys.path.insert(
                    0, str(Path(__file__).resolve().parents[2] / "common-sense/scripts")
                )
                from metax_upgrade.environment import collect_environment
                from metax_upgrade.preparation_gate import kernel_environment

                sources = plan["sources"]
                with kernel_environment(run_dir):
                    report = collect_environment(
                        Path(sources["vllm"]),
                        Path(sources["metax"]),
                        Path(sources["runtime_cwd"]),
                    )
                preparation.verify(run_dir, report)
        except (ValueError, KeyError, OSError, subprocess.SubprocessError) as exc:
            path = run_dir / "PREPARATION.json"
            if (
                args.action in {"sync-sources", "install", "recover", "verify"}
                and path.exists()
            ):
                blocked = json.loads(path.read_text())
                blocked.update(
                    status="blocked", blocker=str(exc), checked_at=preparation.now()
                )
                preparation.write_json(path, blocked)
            raise
        record = json.loads((run_dir / "PREPARATION.json").read_text())
        print(json.dumps(record, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
