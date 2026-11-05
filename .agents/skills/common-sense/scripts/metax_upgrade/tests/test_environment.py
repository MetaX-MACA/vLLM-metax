# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Offline checks for preflight identity, requirements, and output handling."""

import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from metax_upgrade.environment import (
    _binary_identity,
    collect_environment,
    _conda_selection,
    _gpu_identity,
    _sanitize,
    _runtime_environment,
    _source_identity,
    fingerprint,
    read_requirements,
)


class RequirementTests(unittest.TestCase):
    def test_pins_includes_constraints_and_unsupported_forms(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "requirements.txt").write_text(
                "--extra-index-url https://user:secret@example.invalid/simple\n"
                "-r nested.txt\n-c constraints.txt\n"
                "some_pkg==2.0+local\nmissing==1\n"
                "torch>=2; python_version > '3.9'\n"
            )
            (root / "nested.txt").write_text("other.pkg==3\n")
            (root / "constraints.txt").write_text("optional==1\n")
            packages = [
                {"normalized_name": "some-pkg", "version": "2.0+local"},
                {"normalized_name": "other-pkg", "version": "4"},
            ]
            report = read_requirements(root / "requirements.txt", packages)
            statuses = {
                entry.get("normalized_name"): entry["status"]
                for entry in report["entries"]
                if "normalized_name" in entry
            }
            self.assertEqual(
                statuses,
                {
                    "some-pkg": "match",
                    "other-pkg": "mismatch",
                    "missing": "missing",
                    "optional": "constraint_not_installed",
                },
            )
            self.assertEqual(len(report["files"]), 3)
            self.assertTrue(
                any("manual evaluation" in issue for issue in report["issues"])
            )
            self.assertNotIn("user:secret", str(report))

    def test_cycles_missing_files_and_ambiguous_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "requirements.txt"
            path.write_text("-r requirements.txt\n-r absent.txt\nname==1\n")
            packages = [{"normalized_name": "name", "version": "1"}] * 2
            report = read_requirements(path, packages)
            self.assertTrue(any("cycle" in issue for issue in report["issues"]))
            self.assertTrue(any("Cannot read" in issue for issue in report["issues"]))
            self.assertEqual(report["entries"][-1]["status"], "ambiguous")


class IdentityTests(unittest.TestCase):
    def test_same_version_native_replacement_changes_fingerprint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            library = root / "libexample.so.1"
            library.write_bytes(b"old-binary")
            stat = library.stat()

            class Distribution:
                metadata = {"Name": "native_component"}
                version = "1.0"
                files = ["libexample.so.1"]
                record = "libexample.so.1,record-hash,10\n"

                def locate_file(self, path):
                    return root / path

                def read_text(self, name):
                    return self.record if name == "RECORD" else None

            distribution = Distribution()
            packages = [{"normalized_name": "native-component", "version": "1.0"}]
            with patch(
                "metax_upgrade.environment.metadata.distributions",
                return_value=[distribution],
            ):
                before = _binary_identity(packages, {"native_component"})
                replacement = root / "replacement"
                replacement.write_bytes(b"new-binary")
                os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
                os.replace(replacement, library)
                after = _binary_identity(packages, {"native_component"})
                self.assertNotEqual(
                    fingerprint({"identity": before}), fingerprint({"identity": after})
                )
                distribution.record = "updated RECORD content\n"
                changed_record = _binary_identity(packages, {"native_component"})
                self.assertNotEqual(
                    fingerprint({"identity": after}),
                    fingerprint({"identity": changed_record}),
                )
                library.unlink()
                missing = _binary_identity(packages, {"native_component"})
                self.assertTrue(missing["issues"])

    def test_dispatch_environment_allowlist_changes_identity_without_secret_dump(self):
        with patch.dict(
            os.environ,
            {
                "VLLM_PLUGINS": "metax",
                "VLLM_METAX_SUPPORTS_FP8": "0",
                "SERVICE_SECRET": "private",
            },
        ):
            before = _runtime_environment()
            with patch.dict(os.environ, {"VLLM_METAX_SUPPORTS_FP8": "1"}):
                after = _runtime_environment()
        self.assertEqual(before["VLLM_PLUGINS"], "metax")
        self.assertEqual(after["VLLM_METAX_SUPPORTS_FP8"], "1")
        self.assertNotIn("SERVICE_SECRET", before)
        self.assertNotEqual(
            fingerprint({"identity": before}), fingerprint({"identity": after})
        )

    def test_inherited_conda_prefix_does_not_select_parent_for_current_venv(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = str(Path(directory) / "venv")
            parent = str(Path(directory) / "conda")
            selection = _conda_selection(prefix, parent, parent)
            self.assertFalse(selection["is_selected_conda"])
            self.assertFalse(selection["inherited_prefix_matches_selected"])
            self.assertEqual(selection["selected_prefix"], prefix)
            self.assertEqual(selection["inherited_prefix"], parent)
            (Path(parent) / "conda-meta").mkdir(parents=True)
            self.assertTrue(
                _conda_selection(parent, parent, parent)["is_selected_conda"]
            )

    def test_diagnostics_do_not_change_fingerprint(self):
        report = {
            "identity": {"packages": [{"name": "torch", "version": "1"}]},
            "collected_at": "first",
            "details": {"gpu_memory": 1},
        }
        changed = copy.deepcopy(report)
        changed["collected_at"] = "second"
        changed["details"] = {"gpu_memory": 123}
        self.assertEqual(fingerprint(report), fingerprint(changed))
        changed["identity"]["packages"][0]["version"] = "2"
        self.assertNotEqual(fingerprint(report), fingerprint(changed))

    def test_upstream_content_changes_identity_but_metax_edits_do_not(self):
        probe = {
            name: {
                "local": {
                    "checkout": f"/source/{name}",
                    "exists": True,
                    "commit": "old",
                    "manifest_sha256": "old",
                }
            }
            for name in ("vllm", "vllm_metax")
        }
        before = _source_identity(probe)
        probe["vllm_metax"]["local"].update(
            commit="new", manifest_sha256="new", status="M file.py"
        )
        self.assertEqual(before, _source_identity(probe))
        probe["vllm"]["local"]["manifest_sha256"] = "changed"
        self.assertNotEqual(before, _source_identity(probe))

    def test_gpu_identity_excludes_availability(self):
        template = "mx-smi version: 2.3.4\nGPU#0 MXC500 0000:00:09.0 {} (UUID: GPU-a)\n"
        self.assertEqual(
            _gpu_identity(template.format("Available")),
            _gpu_identity(template.format("Unavailable")),
        )
        self.assertEqual(_gpu_identity(template.format("Available"))[0]["index"], 0)
        self.assertEqual(_gpu_identity("GPU 1: MXC550 (UUID: GPU-b)")[0]["index"], 1)


class CollectorTests(unittest.TestCase):
    def test_missing_metax_collector_never_uses_upstream_collector(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metax = root / "metax"
            (metax / "vllm_metax").mkdir(parents=True)
            upstream = root / "vllm/collect_env.py"
            upstream.parent.mkdir()
            upstream.write_text("raise RuntimeError('Must never execute upstream collector')\n")
            result = {"status": "ok", "returncode": 0, "stdout": "{}", "stderr": ""}
            with patch("metax_upgrade.environment._run", return_value=result):
                report = collect_environment(root, metax, root, timeout=1)
            self.assertEqual(report["details"]["collect_env"]["status"], "missing")
            self.assertEqual(report["details"]["collect_env"]["source"], str(metax / "vllm_metax/collect_env.py"))
            self.assertIn("executable", report["identity"]["python"])
            self.assertIn("packages", report["identity"])


class RedactionTests(unittest.TestCase):
    def test_environment_section_credentials_and_known_secret_values(self):
        with patch.dict(os.environ, {"SERVICE_TOKEN": "arbitrary-secret-value"}):
            value = _sanitize(
                "https://user:pass@example.invalid/path arbitrary-secret-value\n"
                "api_key=abc123\nEnvironment Variables\nHOME=/private/path\n"
            )
        for secret in (
            "user:pass",
            "arbitrary-secret-value",
            "abc123",
            "/private/path",
        ):
            self.assertNotIn(secret, value)


if __name__ == "__main__":
    unittest.main()
