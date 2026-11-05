# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Scoped invalidation, mandatory local checks, model audits and baseline coverage."""

import json
from pathlib import Path
import subprocess
import sys
import unittest

from metax_upgrade import acceptance, model_coverage, planning, state as store, workflow
from metax_upgrade.tests import test_upgrade as fixtures
from metax_upgrade.tests import test_workflow as full_fixtures


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.UpgradeTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.approve()
        self.fixture.task("op")
        self.fixture.task("consumer", dependency="op")
        self.fixture.task("unrelated")

    def test_contract_updates_only_touch_participants_and_consumers(self):
        state = store.load(self.fixture.run)
        before = {
            key: (store.source_signature(state, key), store.task_signature(state, key))
            for key in state["tasks"]
        }
        contract = {
            "id": "layout",
            "upstream": "confirmed-target",
            "summary": "cache layout",
            "invariants": ["block-relative indices"],
            "producer": "op",
            "consumers": ["consumer"],
        }
        store.put_contract(state, contract)
        self.assertEqual(
            before["unrelated"],
            (
                store.source_signature(state, "unrelated"),
                store.task_signature(state, "unrelated"),
            ),
        )
        self.assertNotEqual(before["op"][0], store.source_signature(state, "op"))
        self.assertNotEqual(
            before["consumer"][0], store.source_signature(state, "consumer")
        )
        store.put_contract(state, {**contract, "consumers": ["unrelated"]})
        self.assertTrue(state["tasks"]["consumer"].get("revalidation_reason"))
        self.assertTrue(state["tasks"]["unrelated"].get("revalidation_reason"))

    def test_matrix_changes_preserve_local_source_evidence(self):
        self.fixture.check()
        state = store.load(self.fixture.run)
        before = store.source_signature(state, "op")
        final_before = store.task_signature(state, "op")
        acceptance.put_matrix(
            state,
            {
                "rows": [
                    {
                        "id": "model",
                        "status": "supported",
                        "configuration": {"dtype": "bf16"},
                        "tasks": ["op"],
                        "requires": ["e2e"],
                    }
                ]
            },
        )
        self.assertEqual(before, store.source_signature(state, "op"))
        self.assertNotEqual(final_before, store.task_signature(state, "op"))
        self.assertFalse(store.missing_evidence(state, "op"))

    def test_layout_and_index_changes_need_contract_and_component_numerics(self):
        state = store.load(self.fixture.run)
        planning.put_checks(
            state,
            "op",
            {
                "changes": ["layout", "indexing"],
                "reason": "Cache byte stride and index units changed",
                "cases": ["unaligned block", "mixed prefill/decode"],
            },
        )
        self.assertTrue(
            {"static", "review", "contract", "component_numeric"}
            <= set(state["tasks"]["op"]["requires"])
        )
        with self.assertRaisesRegex(ValueError, "matching explicit"):
            planning.put_checks(
                state,
                "op",
                {
                    "changes": ["quantization"],
                    "reason": "FP8",
                    "cases": ["scales"],
                    "defer_numeric": {
                        "feature": "fp8",
                        "chip": "C5xx",
                        "reason": "unsupported",
                    },
                },
            )

    def test_matrix_change_keeps_component_closure_but_invalidates_model_closure(self):
        self.fixture.check()
        state = store.load(self.fixture.run)
        task = state["tasks"]["op"]
        task["requires"].append("closure")
        original = dict(task["evidence"][0])
        component = {
            **original,
            "kind": "closure",
            "execution_scope": "component",
            "signature": store.source_signature(state, "op"),
        }
        model = {
            **original,
            "kind": "closure",
            "execution_scope": "model",
            "signature": store.task_signature(state, "op"),
        }
        task["evidence"].extend([component, model])
        acceptance.put_matrix(
            state,
            {
                "rows": [
                    {
                        "id": "model",
                        "status": "supported",
                        "configuration": {"dtype": "fp16"},
                        "tasks": ["op"],
                        "requires": ["e2e"],
                    }
                ]
            },
        )
        store.invalidate(
            state, {"op"}, "Model matrix changed", kinds=workflow.MODEL_KINDS
        )
        self.assertFalse(component["stale"])
        self.assertTrue(model["stale"])
        self.assertNotIn("closure", store.missing_evidence(state, "op"))

    def test_global_analysis_is_required_before_source_work(self):
        state = store.load(self.fixture.run)
        state["inputs"]["mode"] = "full"
        workflow.initialize(state)
        workflow.register_task(state, "op", "framework")
        with self.assertRaisesRegex(ValueError, "global dependency"):
            workflow.assert_task_start(state, "op")

    def test_declared_dependency_cycles_need_a_coordinated_strategy(self):
        state = store.load(self.fixture.run)
        state["inputs"]["mode"] = "full"
        workflow.initialize(state)
        paths = list(state["inventory"])[:2]
        plan = {
            "files": list(state["inventory"]),
            "models": [],
            "summary": "Synthetic mutually dependent interfaces",
            "edges": [
                {"producer": paths[0], "consumer": paths[1], "contract": "layout"},
                {"producer": paths[1], "consumer": paths[0], "contract": "metadata"},
            ],
            "cycles": [],
        }
        with self.assertRaisesRegex(ValueError, "Dependency cycle"):
            planning.put_analysis(state, plan)
        planning.put_analysis(
            state,
            {
                **plan,
                "cycles": [
                    {
                        "files": paths,
                        "strategy": "Adapt both interfaces as one coordinated source unit",
                    }
                ],
            },
        )

    def test_shared_models_apis_are_allowed_and_indirect_concrete_import_is_rejected(
        self,
    ):
        root = self.fixture.metax
        model = root / "vllm_metax/models/example.py"
        model.parent.mkdir(parents=True)
        model.write_text(
            "from vllm.model_executor.models.interfaces import SupportsPP\nfrom vllm.model_executor.models.utils import AutoWeightsLoader\nfrom vllm_metax.helper import Layer\n"
        )
        helper = root / "vllm_metax/helper.py"
        helper.write_text("from vllm.models.deepseek_v4 import DeepseekV4ForCausalLM\n")
        auditor = (
            Path(model_coverage.__file__).resolve().parents[3]
            / "model-adaptation/scripts/audit_imports.py"
        )
        command = [
            sys.executable,
            "-B",
            str(auditor),
            "--source",
            str(root),
            "--entry",
            "vllm_metax/models/example.py",
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(
            json.loads(result.stdout)["prohibited_imports"][0]["file"],
            "vllm_metax/helper.py",
        )
        helper.write_text("from vllm.config import VllmConfig\n")
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class BaselineCoverageTests(unittest.TestCase):
    def setUp(self):
        self.fixture = full_fixtures.WorkflowTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.cli = self.fixture.cli
        self.run = self.fixture.run
        self.key = "tools/models/example-dummy"
        config = self.fixture.metax / self.key / "config.json"
        config.parent.mkdir(parents=True)
        config.write_text('{"architectures": ["ExampleForCausalLM"]}')
        self.cli("status")
        state = store.load(self.run)
        plan = self.run / "analysis.json"
        plan.write_text(
            json.dumps(
                {
                    "files": list(state["inventory"]),
                    "models": [self.key],
                    "model_contracts": {
                        self.key: {
                            "framework_interfaces": "Synthetic scalar module",
                            "cache_layout": "No cache in fixture",
                            "quantization": "No quantization in fixture",
                        }
                    },
                    "edges": [],
                    "cycles": [],
                    "summary": "Synthetic independent model",
                }
            )
        )
        self.cli("analysis-put", "--file", plan)

    def bind(self):
        state = store.load(self.run)
        row = state["support_matrix"]["models"]
        row["model_sample"] = self.key
        row["configuration"] = {
            "model_path": str(self.fixture.metax / self.key),
            "load_format": "dummy",
            "validation_mode": "dummy_smoke",
        }
        matrix = self.run / "matrix.json"
        matrix.write_text(json.dumps({"rows": list(state["support_matrix"].values())}))
        self.cli("matrix-put", "--file", matrix)
        bindings = self.run / "bindings.json"
        bindings.write_text(
            json.dumps(
                {
                    "models": [
                        {
                            "id": self.key,
                            "spec": self.fixture.files["models"][1],
                            "matrix_rows": ["models"],
                        }
                    ]
                }
            )
        )
        self.cli("model-bind", "--file", bindings)

    def test_unbound_model_sample_blocks_completion(self):
        state = store.load(self.run)
        self.assertIn(self.key, state["model_inventory"])
        self.assertTrue(
            any("not bound" in value for value in model_coverage.finish_issues(state))
        )

    def test_resource_skip_never_waives_local_checks_and_stales_with_evidence(self):
        self.fixture.reach("validation")
        self.bind()
        evidence = self.run / "gpu-inventory.txt"
        evidence.write_text("Synthetic inventory: 1 GPU available, 2 required\n")
        with self.assertRaisesRegex(ValueError, "real GPU-count or VRAM deficit"):
            self.cli(
                "model-skip",
                "--id",
                self.key,
                "--reason-code",
                "insufficient_gpus",
                "--required",
                "1",
                "--available",
                "2",
                "--evidence-file",
                evidence,
                "--note",
                "Not a deficit",
            )
        self.cli(
            "model-skip",
            "--id",
            self.key,
            "--reason-code",
            "insufficient_gpus",
            "--required",
            "2",
            "--available",
            "1",
            "--evidence-file",
            evidence,
            "--note",
            "Synthetic two-rank case cannot fit on one GPU",
        )
        state = store.load(self.run)
        self.assertTrue(model_coverage.valid_skip(state, self.key))
        self.assertFalse(
            {"e2e", "batched_test"} & set(store.missing_evidence(state, "models"))
        )
        for phase in self.fixture.files:
            self.cli("task-set", "--id", phase, "--status", "done")
        self.cli("finish")
        state = store.load(self.run)
        self.assertTrue(state["completed_at"])
        self.assertEqual(state["model_inventory"][self.key]["status"], "skipped")
        evidence.write_text("Resource inventory changed\n")
        self.assertFalse(model_coverage.valid_skip(state, self.key))
        self.assertTrue(model_coverage.finish_issues(state))
        self.assertIn("e2e", store.missing_evidence(state, "models"))

    def test_model_review_is_required_and_spec_changes_close_source_gate(self):
        self.fixture.reach("models")
        with self.assertRaisesRegex(ValueError, "model spec"):
            self.fixture.check("models")
        self.fixture.implement("models")
        state = store.load(self.run)
        self.assertFalse(model_coverage.review_issues(state, "models"))
        (self.fixture.metax / self.fixture.files["models"][1]).write_text(
            "Changed model difference spec\n"
        )
        self.assertTrue(model_coverage.review_issues(state, "models"))
        self.assertFalse(workflow.source_ready(state, "models"))

    def test_every_bound_sample_needs_its_own_current_e2e_and_batched_result(self):
        self.fixture.reach("validation")
        self.bind()
        self.assertEqual(
            self.fixture.check("models", "e2e", "model", self.fixture.report_code()), 0
        )
        self.assertTrue(model_coverage.finish_issues(store.load(self.run)))
        self.assertEqual(
            self.fixture.check(
                "models", "batched_test", "model", self.fixture.report_code()
            ),
            0,
        )
        state = store.load(self.run)
        self.assertFalse(model_coverage.finish_issues(state))
        self.assertEqual(state["model_inventory"][self.key]["status"], "passed")
        (self.fixture.metax / self.key / "config.json").write_text(
            '{"architectures": ["ModifiedModel"]}'
        )
        self.cli("status")
        state = store.load(self.run)
        self.assertTrue(model_coverage.finish_issues(state))
        self.assertEqual(state["model_inventory"][self.key]["status"], "pending")
