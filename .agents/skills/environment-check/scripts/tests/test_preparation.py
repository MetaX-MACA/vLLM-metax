# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Offline preparation checks: real temporary Git/files, no package installation or GPU."""

import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from metax_environment import install, preparation as prep
from metax_upgrade import (
    environment,
    lifecycle,
    preparation_gate,
    state as store,
    upgrade,
)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.metax = self.root / "metax"
        self.vllm = self.root / "vllm"
        self.run = self.root / "run"
        for path in (self.metax, self.vllm):
            (path / "requirements").mkdir(parents=True)
            (path / "requirements/common.txt").write_bytes(b"dependency==2\n")
            (path / "pyproject.toml").write_text("# existing dirty content\n")
            subprocess.run(["git", "init", "-q", "-b", "master", str(path)], check=True)
            subprocess.run(["git", "-C", str(path), "add", "."], check=True)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(path),
                    "-c",
                    "user.name=Test",
                    "-c",
                    "user.email=test@example.invalid",
                    "commit",
                    "-qm",
                    "fixture",
                ],
                check=True,
            )
        (self.metax / "setup.py").write_text(
            'def fixed_version_scheme(version):\n    return "0.32.0"\n\nSUFFIX = "+maca.torch"\n'
        )
        subprocess.run(["git", "-C", str(self.vllm), "tag", "v0.32.0"], check=True)
        patcher = patch.object(prep, "PROJECT_ROOT", self.metax)
        patcher.start()
        self.addCleanup(patcher.stop)
        private = [
            {
                "normalized_name": name,
                "version": "2+metax3",
                "installation_root": sys.prefix,
            }
            for name in ("torch", "triton")
        ]
        private.append(
            {
                "normalized_name": "mcoplib",
                "version": "0.4.14",
                "installation_root": sys.prefix,
            }
        )
        self.report = {
            "identity": {
                "python": {
                    "executable": sys.executable,
                    "prefix": sys.prefix,
                    "base_prefix": sys.base_prefix,
                    "version": sys.version,
                },
                "sdk": {"version": "3"},
                "gpu": {"models": [{"description": "C500"}]},
                "system": {"machine": "x86_64"},
                "packages": private,
                "runtime_environment": {"USE_PRECOMPILED_KERNEL": "1"},
                "native_files": {"distributions": []},
            },
            "details": {
                "requirements": {"entries": [{"normalized_name": "mcoplib"}]},
                "source_probe": {},
            },
            "readiness": {"issues": []},
        }
        for module, source in (("vllm", self.vllm), ("vllm_metax", self.metax)):
            self.report["details"]["source_probe"][module] = {
                "effective_import_probe": {
                    "includes_local_package": module == "vllm_metax"
                },
                "installed_distributions": [
                    {
                        "version": "0.32.0+empty"
                        if module == "vllm"
                        else "0.32.0+metax3",
                        "installation_root": sys.prefix,
                        "provenance": {
                            "local_source_or_archive": str(source),
                            "editable": module == "vllm_metax",
                        },
                        "package_trees": [
                            {
                                "selected_by_import_probe": True,
                                "comparison_to_local": {"status": "identical"},
                            }
                        ],
                    }
                ],
            }
        self.plan = {
            "schema_version": 1,
            "kernel_provider": "mcoplib",
            "target": {
                "ref": "v0.32.0",
                "commit": prep.git(self.vllm, "rev-parse", "HEAD"),
                "version": "0.32.0",
                "previous_release": "v0.31.0",
                "history": "previous-release-range",
            },
            "python": {
                **self.report["identity"]["python"],
                "base_executable": sys.executable,
                "selection": "detected_uv",
                "system_site_packages": False,
            },
            "sources": {
                "vllm": str(self.vllm),
                "metax": str(self.metax),
                "runtime_cwd": str(self.metax),
                "metax_branch": "master",
                "metax_commit": prep.git(self.metax, "rev-parse", "HEAD"),
            },
            "environment_choice": "reuse",
            "installs": {"vllm": False, "metax": False},
            "private_packages": private,
            "private_native_identity": [],
            "host_identity": {
                key: self.report["identity"][key] for key in ("sdk", "gpu", "system")
            },
            "git_policy": "preserve-worktree",
            "ordinary_dependencies": "allow-with-private-locks",
            "known_issues": [],
            "validation_scope": {"chip": "C500", "dtypes": ["bf16", "fp16", "int8"]},
        }

    def approved(self):
        prep.put_plan(self.run, self.plan)
        prep.approve(
            self.run,
            prep.digest(self.plan),
            "Approve the displayed complete preparation plan.",
        )

    def inputs(self):
        return {
            "vllm_source": str(self.vllm),
            "metax_source": str(self.metax),
            "runtime_cwd": str(self.metax),
            "mode": "full",
        }

    def test_kernel_provider_is_explicit_and_source_mode_requires_rebuild(self):
        for provider in (None, "auto"):
            plan = {**self.plan, "kernel_provider": provider}
            with self.assertRaisesRegex(ValueError, "Explicitly confirm"):
                prep.validate_plan(plan)
        plan = {**self.plan, "kernel_provider": "source"}
        with self.assertRaisesRegex(ValueError, "rebuild"):
            prep.validate_plan(plan)
        plan["installs"] = {"vllm": False, "metax": True}
        prep.validate_plan(plan)

    def test_mcoplib_mode_requires_the_existing_image_package(self):
        self.plan["private_packages"] = [
            row
            for row in self.plan["private_packages"]
            if row["normalized_name"] != "mcoplib"
        ]
        with self.assertRaisesRegex(ValueError, "image mcoplib"):
            prep.validate_plan(self.plan)

    def test_source_mode_verifies_effective_runtime_flag(self):
        self.plan["kernel_provider"] = "source"
        self.plan["installs"]["metax"] = True
        with self.assertRaisesRegex(ValueError, "Runtime USE_PRECOMPILED_KERNEL"):
            prep.verify_report(self.plan, self.report)
        self.report["identity"]["runtime_environment"]["USE_PRECOMPILED_KERNEL"] = "0"
        prep.verify_report(self.plan, self.report)

    def test_source_mode_cannot_accept_a_precompiled_install_receipt(self):
        self.plan["kernel_provider"] = "source"
        self.plan["installs"]["metax"] = True
        self.report["identity"]["runtime_environment"]["USE_PRECOMPILED_KERNEL"] = "0"
        self.approved()
        prep.write_json(
            self.run / "preparation/install-metax.json",
            {
                "status": "installed",
                "plan_digest": prep.digest(self.plan),
                "kernel_provider": "mcoplib",
                "kernel_environment": {"USE_PRECOMPILED_KERNEL": "1"},
            },
        )
        with self.assertRaisesRegex(ValueError, "did not succeed"):
            prep.verify(self.run, self.report)

    def test_selected_source_flag_overrides_parent_and_context_restores_it(self):
        self.plan["kernel_provider"] = "source"
        self.plan["installs"]["metax"] = True
        self.approved()
        with patch.dict(os.environ, {"USE_PRECOMPILED_KERNEL": "1"}):
            with preparation_gate.kernel_environment(self.run):
                self.assertEqual(os.environ["USE_PRECOMPILED_KERNEL"], "0")
            self.assertEqual(os.environ["USE_PRECOMPILED_KERNEL"], "1")
            state = {"inputs": self.inputs(), "run_dir": str(self.run)}
            with patch.object(
                environment,
                "collect_environment",
                side_effect=lambda *a, **k: os.environ["USE_PRECOMPILED_KERNEL"],
            ):
                self.assertEqual(upgrade.collect(state), "0")
            self.assertEqual(os.environ["USE_PRECOMPILED_KERNEL"], "1")

    def test_malformed_kernel_backend_is_rejected_as_invalid_evidence(self):
        for malformed in (None, [], "source"):
            self.assertIsNotNone(
                prep.backend_issue(self.plan, {"kernel_backend": malformed})
            )

    def test_source_production_origins_reject_indirect_mcoplib_import(self):
        self.plan["kernel_provider"] = "source"
        self.plan["installs"]["metax"] = True
        self.approved()
        backend = {
            "provider": "source",
            "use_precompiled_kernel": "0",
            "loaded_mcoplib_modules": [],
        }
        self.assertIsNone(prep.backend_issue(self.plan, {"kernel_backend": backend}))
        backend["loaded_mcoplib_modules"] = ["mcoplib.triton_gumbel_sample"]
        origins = self.root / "origins.json"
        origins.write_text(json.dumps({"kernel_backend": backend}))
        valid, reason = upgrade.origins_valid(
            {"run_dir": str(self.run)}, "e2e", origins
        )
        self.assertFalse(valid)
        self.assertIn("imported mcoplib", reason)

    def test_source_flag_reaches_a_real_check_child_with_conflicting_parent_flag(self):
        from metax_upgrade.tests.test_upgrade import synthetic_mapping

        self.plan["kernel_provider"] = "source"
        self.plan["installs"]["metax"] = True
        self.report["identity"]["runtime_environment"]["USE_PRECOMPILED_KERNEL"] = "0"
        self.approved()
        prep.write_json(
            self.run / "preparation/install-metax.json",
            {
                "status": "installed",
                "plan_digest": prep.digest(self.plan),
                "kernel_provider": "source",
                "kernel_environment": {"USE_PRECOMPILED_KERNEL": "0"},
            },
        )
        prep.verify(self.run, self.report)

        def cli(action, *args):
            return upgrade.main([action, "--run-dir", str(self.run), *map(str, args)])

        with (
            patch.object(upgrade, "collect", return_value=self.report),
            contextlib.redirect_stdout(io.StringIO()),
            patch.dict(os.environ, {"USE_PRECOMPILED_KERNEL": "1"}),
        ):
            cli(
                "init",
                "--vllm-source",
                self.vllm,
                "--metax-source",
                self.metax,
                "--runtime-cwd",
                self.metax,
                "--scope",
                "source flag inheritance",
                "--mode",
                "component",
            )
            cli(
                "task-add",
                "--id",
                "native-flag",
                "--title",
                "Verify source flag",
                "--files",
                "setup.py",
                "--requires",
                "static",
                "--configuration",
                "source",
                "--next-action",
                "Check actual child environment",
                "--owner",
                "csrc-upgrade",
            )
            synthetic_mapping(self.run, "native-flag")
            checks = self.root / "checks.json"
            checks.write_text(
                json.dumps(
                    {
                        "changes": ["none"],
                        "reason": "Offline environment propagation check",
                    }
                )
            )
            cli("checks-put", "--id", "native-flag", "--file", checks)
            self.assertEqual(
                cli(
                    "run-check",
                    "--id",
                    "native-flag",
                    "--kind",
                    "static",
                    "--configuration",
                    "source",
                    "--",
                    sys.executable,
                    "-c",
                    "import os; assert os.environ['USE_PRECOMPILED_KERNEL'] == '0'",
                ),
                0,
            )
            self.assertEqual(os.environ["USE_PRECOMPILED_KERNEL"], "1")

    def test_requires_target_and_current_owning_project(self):
        for key, value in (
            ("ref", ""),
            ("commit", "master"),
            ("version", "0.32.0+empty"),
        ):
            plan = copy.deepcopy(self.plan)
            plan["target"][key] = value
            with self.assertRaises(ValueError):
                prep.validate_plan(plan)
        self.plan["sources"]["metax"] = str(self.root / "other-project")
        with self.assertRaisesRegex(ValueError, "owning"):
            prep.validate_plan(self.plan)

    def test_requires_actual_reply_and_completed_installation(self):
        prep.put_plan(self.run, self.plan)
        with self.assertRaises(ValueError):
            prep.approve(self.run, prep.digest(self.plan), " ")
        with self.assertRaises(ValueError):
            prep.require_ready(
                self.run, self.inputs(), prep.digest(self.report["identity"])
            )
        prep.approve(self.run, prep.digest(self.plan), "Explicit approval")
        with self.assertRaisesRegex(ValueError, "not completed"):
            prep.require_ready(
                self.run, self.inputs(), prep.digest(self.report["identity"])
            )

    def test_conditional_choices_preserve_displayed_proposal_and_actual_reply(self):
        prep.put_plan(self.run, self.plan)
        prep.approve(
            self.run,
            prep.digest(self.plan),
            "Use the supplied v0.32.0 tag and existing uv path.",
            proposal="Select a target tag and uv path; resolve and freeze that tag before installation.",
        )
        record, _ = prep.load_plan(self.run)
        approval = json.loads((self.run / record["approval"]).read_text())
        self.assertIn("v0.32.0", approval["reply"])
        self.assertIn("resolve and freeze", approval["displayed_proposal"])
        self.assertEqual(approval["plan_digest"], prep.digest(self.plan))

    def test_machine_verification_keeps_original_reply_and_reuses_approval(self):
        self.approved()
        record = prep.verify(self.run, self.report)
        approved = (self.run / record["approval"]).read_bytes()
        confirmation = prep.require_ready(
            self.run, self.inputs(), environment.fingerprint(self.report)
        )
        self.assertEqual(confirmation["mode"], "preparation-plan")
        self.assertNotIn("reply", confirmation)
        prep.verify(self.run, self.report)
        self.assertEqual(approved, (self.run / record["approval"]).read_bytes())
        with self.assertRaisesRegex(ValueError, "already has an approval"):
            prep.approve(self.run, prep.digest(self.plan), "Replacement reply")

    def test_private_package_without_vendor_suffix_is_locked(self):
        report = copy.deepcopy(self.report)
        report["details"]["requirements"]["entries"] = [{"normalized_name": "mcoplib"}]
        self.assertIn(
            "mcoplib", {row["normalized_name"] for row in prep.private_locks(report)}
        )
        with self.assertRaisesRegex(ValueError, "Unapproved private"):
            plan = {
                **self.plan,
                "private_packages": [
                    row
                    for row in self.plan["private_packages"]
                    if row["normalized_name"] != "mcoplib"
                ],
            }
            prep.verify_identity(plan, report)

    def test_private_replacement_host_drift_and_native_replacement_are_blocked(self):
        for change in ("private", "native", "host"):
            report = copy.deepcopy(self.report)
            if change == "private":
                report["identity"]["packages"][0]["version"] = "public-build"
            elif change == "native":
                report["identity"]["native_files"]["distributions"].append(
                    {"normalized_name": "torch", "inode": 123}
                )
            else:
                report["identity"]["gpu"] = {"models": ["other-chip"]}
            with self.assertRaises(ValueError):
                prep.verify_report(self.plan, report)

    def test_wrong_model_package_origin_noneditable_and_missing_install_block(self):
        for change in ("origin", "editable", "missing"):
            report = copy.deepcopy(self.report)
            package = report["details"]["source_probe"]["vllm_metax"]
            if change == "origin":
                package["effective_import_probe"]["includes_local_package"] = False
            elif change == "editable":
                package["installed_distributions"][0]["provenance"]["editable"] = False
            else:
                package["installed_distributions"] = []
            with self.assertRaises(ValueError):
                prep.verify_report(self.plan, report)

    def test_required_reinstallation_needs_successful_receipts(self):
        self.plan["environment_choice"] = "reinstall"
        self.plan["installs"] = {"vllm": True, "metax": True}
        self.approved()
        with self.assertRaises(OSError):
            prep.verify(self.run, self.report)
        self.assertEqual(
            json.loads((self.run / "PREPARATION.json").read_text())["status"], "blocked"
        )
        for package in ("vllm", "metax"):
            prep.write_json(
                self.run / "preparation" / ("install-" + package + ".json"),
                {"status": "failed", "plan_digest": prep.digest(self.plan)},
            )
        with self.assertRaisesRegex(ValueError, "did not succeed"):
            prep.verify(self.run, self.report)

    def test_approved_plan_and_verified_snapshot_tampering_is_rejected(self):
        self.approved()
        record = prep.verify(self.run, self.report)
        path = self.run / record["verification"]
        verified = json.loads(path.read_text())
        verified["environment_fingerprint"] = "other"
        prep.write_json(path, verified)
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            prep.require_ready(
                self.run, self.inputs(), environment.fingerprint(self.report)
            )

    def test_sync_only_common_and_base_version_preserving_suffix_and_other_requirements(
        self,
    ):
        self.approved()
        (self.metax / "requirements/common.txt").write_bytes(b"old==1\n")
        private = self.metax / "requirements/maca_private.txt"
        private.write_bytes(b"# user edit\nmcoplib==1\n")
        setup = self.metax / "setup.py"
        setup.write_text(setup.read_text().replace('"0.32.0"', '"0.31.0"'))
        before = setup.read_text()
        install.sync_sources(self.run)
        self.assertEqual(
            (self.metax / "requirements/common.txt").read_bytes(), b"dependency==2\n"
        )
        self.assertEqual(setup.read_text(), before.replace('"0.31.0"', '"0.32.0"'))
        self.assertEqual(private.read_bytes(), b"# user edit\nmcoplib==1\n")

    def test_temporary_cleaner_changes_restore_on_success_and_failure(self):
        common = self.metax / "requirements/common.txt"
        project = self.metax / "pyproject.toml"
        project.write_bytes(b"# user uncommitted edit\r\n")
        project.chmod(0o640)
        for fail in (False, True):
            before = {path: path.read_bytes() for path in (common, project)}
            try:
                with install.temporary_requirements(
                    self.metax, self.run / "journal.json"
                ):
                    common.write_bytes(b"temporary cleaner contents")
                    project.write_bytes(b"temporary build contents")
                    if fail:
                        raise RuntimeError("failed installer")
            except RuntimeError:
                pass
            self.assertEqual({path: path.read_bytes() for path in before}, before)
            self.assertEqual(project.stat().st_mode & 0o777, 0o640)

    def test_repeating_recovery_preserves_later_source_edits(self):
        journal = self.run / "journal.json"
        project = self.metax / "pyproject.toml"
        with install.temporary_requirements(self.metax, journal):
            project.write_bytes(b"temporarily suppressed dependencies\n")
        project.write_bytes(b"later intentional source edit\n")
        install.restore(journal)
        self.assertEqual(project.read_bytes(), b"later intentional source edit\n")

    def test_interrupted_journal_closes_gate_until_recovered(self):
        self.approved()
        journal = self.run / "preparation/temporary-requirements.json"
        prep.write_json(
            journal, {"source": str(self.metax), "status": "temporary", "files": {}}
        )
        with self.assertRaisesRegex(ValueError, "recovered"):
            prep.verify(self.run, self.report)
        install.restore(journal)
        prep.verify(self.run, self.report)

    def test_full_upgrade_cannot_use_snapshot_only_approval(self):
        state = {"inputs": self.inputs()}
        with self.assertRaisesRegex(ValueError, "environment-check"):
            preparation_gate.confirmation(
                self.run, state, environment.fingerprint(self.report)
            )

    def test_new_component_run_also_requires_the_preparation_plan(self):
        with (
            patch.object(upgrade, "collect", return_value=self.report),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            upgrade.main(
                [
                    "init",
                    "--run-dir",
                    str(self.run),
                    "--vllm-source",
                    str(self.vllm),
                    "--metax-source",
                    str(self.metax),
                    "--runtime-cwd",
                    str(self.metax),
                    "--scope",
                    "new component environment gate",
                    "--mode",
                    "component",
                ]
            )
        state = store.load(self.run)
        self.assertTrue(preparation_gate.uses_plan(self.run, state))
        reply = self.root / "reply.txt"
        reply.write_text("Snapshot approval alone must not bypass installation")
        with (
            self.assertRaisesRegex(ValueError, "environment-check"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            upgrade.main(
                [
                    "confirm",
                    "--run-dir",
                    str(self.run),
                    "--fingerprint",
                    environment.fingerprint(self.report),
                    "--user-response-file",
                    str(reply),
                ]
            )

    def test_full_upgrade_gate_uses_plan_without_second_user_reply(self):
        self.approved()
        prep.verify(self.run, self.report)
        state = {
            "schema_version": store.SCHEMA_VERSION,
            "policy_identity": lifecycle.runtime_identity(),
            "inputs": self.inputs(),
            "tasks": {},
            "environment": {},
        }
        upgrade.snapshot(self.run, state, self.report)
        with patch.object(upgrade, "collect", return_value=copy.deepcopy(self.report)):
            upgrade.assert_ready(self.run, state)
        self.assertEqual(
            state["environment"]["confirmation"]["mode"], "preparation-plan"
        )
        self.assertFalse(list(self.run.glob("environment/approval-*")))

    def test_no_pip_uses_uv_with_selected_interpreter(self):
        with (
            patch.object(
                subprocess, "run", return_value=subprocess.CompletedProcess([], 1)
            ),
            patch.object(install.shutil, "which", return_value="/bin/uv"),
        ):
            command = install.pip_command(
                self.plan, "install", ".", "--no-build-isolation"
            )
        self.assertEqual(
            command[:5], ["/bin/uv", "pip", "install", "--python", sys.executable]
        )

    def test_source_egg_info_does_not_hide_actual_editable_installation(self):
        report = copy.deepcopy(self.report)
        report["details"]["source_probe"]["vllm_metax"][
            "installed_distributions"
        ].append({"metadata_paths": [str(self.metax / "vllm_metax.egg-info")]})
        prep.verify_report(self.plan, report)

    def test_failed_metax_install_restores_dirty_requirements_and_closes_gate(self):
        self.plan["installs"]["metax"] = True
        self.plan["kernel_provider"] = "source"
        self.report["identity"]["runtime_environment"]["USE_PRECOMPILED_KERNEL"] = "0"
        self.approved()
        project = self.metax / "pyproject.toml"
        original = project.read_bytes()

        def run(command, **kwargs):
            self.assertEqual(kwargs["env"]["VLLM_PLUGINS"], "")
            self.assertEqual(kwargs["env"]["USE_PRECOMPILED_KERNEL"], "0")
            self.assertNotIn("VLLM_TARGET_DEVICE", kwargs["env"])
            if "use_existing_metax.py" in command:
                project.write_bytes(b"temporarily suppressed dependencies\n")
                return subprocess.CompletedProcess(command, 0)
            self.assertIn("--constraint", command)
            raise subprocess.CalledProcessError(1, command)

        with (
            patch.object(prep, "check_sources"),
            patch.object(environment, "collect_environment", return_value=self.report),
            patch.object(
                install,
                "pip_command",
                side_effect=lambda _p, op, *args: ["fake-pip", op, *args],
            ),
            patch.object(install.subprocess, "run", side_effect=run),
        ):
            with self.assertRaises(subprocess.CalledProcessError):
                install.install(self.run, "metax")
        self.assertEqual(project.read_bytes(), original)
        record, _ = prep.load_plan(self.run)
        self.assertEqual(record["status"], "blocked")
        receipt = json.loads((self.run / "preparation/install-metax.json").read_text())
        self.assertEqual(receipt["status"], "failed")
        with self.assertRaises(ValueError):
            prep.require_ready(
                self.run, self.inputs(), environment.fingerprint(self.report)
            )

    def test_upstream_empty_device_is_confined_to_its_install(self):
        self.plan["installs"] = {"vllm": True, "metax": True}
        self.approved()
        calls = []

        def run(command, **kwargs):
            calls.append((list(command), dict(kwargs["env"])))
            return subprocess.CompletedProcess(command, 0)

        with (
            patch.object(prep, "check_sources"),
            patch.object(environment, "collect_environment", return_value=self.report),
            patch.object(
                install,
                "pip_command",
                side_effect=lambda _p, op, *args: ["fake-pip", op, *args],
            ),
            patch.object(install.subprocess, "run", side_effect=run),
        ):
            install.install(self.run, "vllm")
            install.install(self.run, "metax")
        self.assertEqual(calls[0][1]["VLLM_TARGET_DEVICE"], "empty")
        self.assertIn("--no-deps", calls[0][0])
        self.assertTrue(all("VLLM_TARGET_DEVICE" not in env for _, env in calls[1:]))
        self.assertIn("use_existing_metax.py", calls[1][0])
        self.assertIn("-e", calls[2][0])

    def test_cli_help_from_foreign_cwd(self):
        script = Path(prep.__file__).resolve().parents[1] / "prepare_environment.py"
        result = subprocess.run(
            [sys.executable, "-B", str(script), "--help"],
            cwd=self.root,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("verify", result.stdout)


if __name__ == "__main__":
    unittest.main()
