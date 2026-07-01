#!/usr/bin/env python3
"""Compare two MACA runtime environment reports and emit a compact JSON diff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in sorted(value.items()):
            flatten(f"{prefix}.{key}" if prefix else str(key), child, out)
    else:
        out[prefix] = value


def diff(left: dict[str, Any], right: dict[str, Any]) -> dict[str, object]:
    a: dict[str, Any] = {}
    b: dict[str, Any] = {}
    flatten("", left, a)
    flatten("", right, b)
    keys = sorted(set(a) | set(b))
    changed = [
        {"key": key, "before": a.get(key), "after": b.get(key)}
        for key in keys
        if a.get(key) != b.get(key)
    ]
    return {"changed_count": len(changed), "changed": changed}


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def self_test() -> None:
    data = diff({"env": {"MACA_HOME": "/a"}}, {"env": {"MACA_HOME": "/b"}})
    if data["changed_count"] != 1:
        raise RuntimeError("self-test failed: expected one changed key")
    print(json.dumps({"ok": True, "changed_count": data["changed_count"]}, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", nargs="?")
    parser.add_argument("after", nargs="?")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.before or not args.after:
        parser.error("before and after are required unless --self-test is used")
    print(json.dumps(diff(load(Path(args.before)), load(Path(args.after))), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
