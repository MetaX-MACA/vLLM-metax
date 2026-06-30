#!/usr/bin/env python3
"""Generate a resource-aware validation plan for MACA containers."""

from __future__ import annotations

import argparse
import itertools
import json

CASES = ['qwen-smoke', 'llama-smoke', 'embedding-smoke']
RESOURCES = {'memory': ['16GB', '64GB'], 'parallel': ['tp1', 'tp2'], 'mode': ['offline', 'server']}


def plan(cases: list[str]) -> list[dict[str, str]]:
    keys = list(RESOURCES)
    rows: list[dict[str, str]] = []
    for case in cases:
        for values in itertools.product(*(RESOURCES[k] for k in keys)):
            row = {"case": case}
            row.update(dict(zip(keys, values)))
            rows.append(row)
    return rows


def self_test() -> None:
    rows = plan(CASES[:1])
    assert rows and rows[0]["case"] == CASES[0]
    print(json.dumps({"ok": True, "rows": len(rows)}, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    print(json.dumps(plan(args.case or CASES), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
