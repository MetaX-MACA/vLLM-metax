#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Tests for branch-port audit completeness checks."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import check_audit_completeness


class AuditCompletenessTest(unittest.TestCase):
    def test_reports_missing_commit(self) -> None:
        first = "a" * 40
        second = "b" * 40
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory = root / "inventory.json"
            report = root / "AUDIT.md"
            inventory.write_text(
                json.dumps(
                    {
                        "commit_count": 2,
                        "commits": [
                            {"sha": first, "subject": "first"},
                            {"sha": second, "subject": "second"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            report.write_text(f"| {first} | fixed | done |\n", encoding="utf-8")

            missing = check_audit_completeness.missing_commits(inventory, report)

            self.assertEqual(missing, [{"sha": second, "subject": "second"}])

    def test_accepts_one_row_per_commit(self) -> None:
        first = "a" * 40
        second = "b" * 40
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inventory = root / "inventory.json"
            report = root / "AUDIT.md"
            inventory.write_text(
                json.dumps(
                    {
                        "commit_count": 2,
                        "commits": [{"sha": first}, {"sha": second}],
                    }
                ),
                encoding="utf-8",
            )
            report.write_text(f"{first}\n{second}\n", encoding="utf-8")

            self.assertEqual(
                check_audit_completeness.missing_commits(inventory, report),
                [],
            )

    def test_requires_logic_audit_and_example_for_port(self) -> None:
        sha = "a" * 40
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "AUDIT.md"
            report.write_text(
                f"## Commit logic audit: {sha}\n\n### Original intent\n\nfixed\n",
                encoding="utf-8",
            )

            self.assertEqual(
                check_audit_completeness.invalid_port_audits(report, [sha]),
                [f"{sha}: missing explanation example"],
            )

    def test_accepts_complete_port_logic_audit(self) -> None:
        sha = "a" * 40
        with tempfile.TemporaryDirectory() as temporary:
            report = Path(temporary) / "AUDIT.md"
            report.write_text(
                f"## Commit logic audit: {sha}\n\n"
                "### Explanation example\n\n"
                "When the input is negative, the adapted logic should return zero.\n",
                encoding="utf-8",
            )

            self.assertEqual(
                check_audit_completeness.invalid_port_audits(report, [sha]),
                [],
            )


if __name__ == "__main__":
    unittest.main()
