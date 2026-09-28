#!/usr/bin/env python3
"""Summarize service, benchmark, or distributed runtime logs into JSON."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

METRICS = ['latency', 'throughput', 'tokens']
ERROR_RE = re.compile(r"(error|failed|timeout|traceback|exception|segmentation fault|core dumped)", re.I)
NUMBER_RE = re.compile(r"(?P<key>[A-Za-z_/-]+)\s*[:=]\s*(?P<value>[0-9]+(?:\.[0-9]+)?)")


def parse(path: Path) -> dict[str, object]:
    errors: list[dict[str, object]] = []
    values: list[dict[str, object]] = []
    with path.open(encoding="utf-8", errors="replace") as log_file:
        for lineno, line in enumerate(log_file, 1):
            if ERROR_RE.search(line):
                errors.append({"line": lineno, "text": line.strip()})
            for match in NUMBER_RE.finditer(line):
                key = match.group("key").lower()
                if not METRICS or any(token in key for token in METRICS):
                    values.append({"line": lineno, "metric": key, "value": float(match.group("value"))})
    return {"path": str(path), "metric_count": len(values), "error_count": len(errors), "metrics": values, "errors": errors}


def self_test() -> None:
    sample = Path("_log_summary_sample.log")
    sample.write_text("latency_ms=12.5\nERROR timeout\n", encoding="utf-8")
    try:
        data = parse(sample)
        if data["metric_count"] != 1 or data["error_count"] != 1:
            raise RuntimeError("self-test failed: unexpected log summary")
        print(json.dumps({"ok": True, "metric_count": data["metric_count"]}, ensure_ascii=False))
    finally:
        sample.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="*")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.logs:
        parser.error("at least one log path is required unless --self-test is used")
    paths = [Path(p) for p in args.logs]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        print(f"Missing log file(s): {', '.join(missing)}", file=sys.stderr)
        return 2
    print(json.dumps([parse(path) for path in paths], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
