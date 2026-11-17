#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Tests for the read-only branch-port inventory."""

from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

import inventory_commits


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
    ).strip()


def commit(repo: Path, name: str, content: str, message: str) -> str:
    (repo / name).write_text(content, encoding="utf-8")
    git(repo, "add", name)
    git(repo, "commit", "-m", message)
    return git(repo, "rev-parse", "HEAD")


class InventoryTest(unittest.TestCase):
    def test_lists_source_only_non_merge_commits(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            git(repo, "init", "-b", "main")
            git(repo, "config", "user.name", "Original Author")
            git(repo, "config", "user.email", "author@example.com")

            commit(repo, "base.txt", "base\n", "base")
            git(repo, "branch", "target")
            git(repo, "checkout", "-b", "source")
            fix_sha = commit(
                repo,
                "fix.txt",
                "fixed\n",
                "casual fix\n\nChange-Id: I-old-source-id\n",
            )

            git(repo, "checkout", "-b", "feature", "HEAD~1")
            perf_sha = commit(
                repo,
                "perf.txt",
                "faster\n",
                "perf(attention): avoid gather\n",
            )
            git(repo, "checkout", "source")
            git(repo, "merge", "--no-ff", "feature", "-m", "Merge feature")
            source_sha = git(repo, "rev-parse", "HEAD")

            git(repo, "checkout", "target")
            target_sha = commit(
                repo,
                "target.txt",
                "new target\n",
                "feat(target): target only\n",
            )

            result = inventory_commits.inventory(repo, "source", "target")

            self.assertEqual(result["source"]["sha"], source_sha)
            self.assertEqual(result["target"]["sha"], target_sha)
            self.assertEqual(result["commit_count"], 2)
            self.assertEqual(result["ignored_merge_count"], 1)
            self.assertEqual(
                [item["sha"] for item in result["commits"]],
                [fix_sha, perf_sha],
            )
            self.assertEqual(result["commits"][0]["subject"], "casual fix")
            self.assertEqual(
                result["commits"][0]["source_change_id"],
                "I-old-source-id",
            )
            self.assertEqual(
                result["commits"][0]["author"]["name"],
                "Original Author",
            )
            self.assertTrue(result["commits"][0]["patch_id"])

    def test_rejects_same_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            git(repo, "init", "-b", "main")
            git(repo, "config", "user.name", "Test")
            git(repo, "config", "user.email", "test@example.com")
            commit(repo, "file.txt", "content\n", "initial")

            with self.assertRaises(inventory_commits.GitError):
                inventory_commits.inventory(repo, "HEAD", "main")


if __name__ == "__main__":
    unittest.main()
