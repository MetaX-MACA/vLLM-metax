# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
"""Path resolution must be independent of process working directory."""
import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from tools.batched_test.launch import Scheduler, SchedulerArgs
from tools.batched_test.model_worker import InferenceTask
from tools.batched_test.paths import TOOL_ROOT, normalize_config, resolve_path
from tools.batched_test.specs import ModelSpec, BenchmarkSpec
from tools.batched_test.suites import load_cases, plan_inference


@pytest.mark.parametrize("cwd_kind", ["repo", "tool", "external"])
def test_bundled_configs_and_cases_from_each_cwd(monkeypatch, tmp_path, cwd_kind):
    cwd = {"repo": TOOL_ROOT.parents[1], "tool": TOOL_ROOT, "external": tmp_path}[cwd_kind]
    monkeypatch.chdir(cwd)
    args = SchedulerArgs(work_dir="results", model_config="configs/model.yaml",
                         text_case="configs/inference/text_case.yaml",
                         image_case="configs/inference/image_case.yaml", dry_run=True)
    scheduler = Scheduler(args)
    assert args.model_config == str(TOOL_ROOT / "configs/model.yaml")
    assert args.work_dir == str(cwd / "results")
    for cfg in scheduler.model_list:
        spec = BenchmarkSpec.parse(cfg.get("benchmark"))
        assert spec.parameters
        parsed = ModelSpec.parse(cfg)
        if "--chat-template" in parsed.serve.extra_args:
            value = parsed.serve.extra_args[parsed.serve.extra_args.index("--chat-template") + 1]
            assert Path(value).is_absolute() and Path(value).is_file()
    images = load_cases(args.image_case, "single-image")
    assert images[0].input.startswith("https://")
    assert all(Path(case.input).is_file() for case in images[1:])


def test_external_config_relative_inputs_and_dump_roundtrip(monkeypatch, tmp_path):
    config_dir = tmp_path / "project"
    config_dir.mkdir()
    (config_dir / "models/local").mkdir(parents=True)
    (config_dir / "template.jinja").write_text("{{ messages }}")
    (config_dir / "params.json").write_text(json.dumps([{"dataset_path": "./data.json", "max_concurrency": 1}]))
    config = config_dir / "models.yaml"
    config.write_text(yaml.safe_dump([{
        "name": "local", "model_path": "./models/local", "infer_type": ["text-only"],
        "serve_config": {"extra_args": {"--chat-template": "./template.jinja"}},
        "benchmark": {"bench_param": "./params.json"},
    }]))
    cases = config_dir / "cases.yaml"
    cases.write_text("- question: hello\n  keywords: [hello]\n")
    images = config_dir / "images.yaml"
    images.write_text("- picture_url: ./pictures/a.png\n  keywords: [cat]\n")
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    monkeypatch.chdir(outside)
    args = SchedulerArgs(work_dir="results", model_config="../project/models.yaml",
                         text_case="../project/cases.yaml", image_case=str(images),
                         dump_selected="selected.yaml", dry_run=True)
    scheduler = Scheduler(args)
    original = ModelSpec.parse(scheduler.model_list[0])
    assert original.model_path == str(config_dir / "models/local")
    assert original.serve.extra_args == ("--chat-template", str(config_dir / "template.jinja"))
    bench = BenchmarkSpec.parse(scheduler.model_list[0]["benchmark"])
    assert bench.parameter_dicts()[0]["dataset_path"] == str(config_dir / "data.json")
    assert load_cases(str(images), "single-image")[0].input == str(config_dir / "pictures/a.png")
    dumped = outside / "selected.yaml"
    assert dumped.exists()
    monkeypatch.chdir(tmp_path)
    reloaded = Scheduler(SchedulerArgs(work_dir="again", model_config=str(dumped),
                                      text_case=str(cases), image_case=str(images), dry_run=True))
    assert ModelSpec.parse(reloaded.model_list[0]) == original
    assert BenchmarkSpec.parse(reloaded.model_list[0]["benchmark"]) == bench


@pytest.mark.parametrize("extra", [
    {"--chat-template": "./template file.jinja", "--tokenizer": "org/tokenizer"},
    ["--chat-template", "./template file.jinja", "--tokenizer", "org/tokenizer"],
    '--chat-template="./template file.jinja" --tokenizer org/tokenizer',
])
def test_extra_args_formats_and_hub_ids(tmp_path, extra):
    raw = {"name": "m", "model_path": "org/model", "serve_config": {"extra_args": extra}}
    spec = ModelSpec.parse(raw, base_dir=tmp_path)
    assert spec.model_path == "org/model"
    assert str(tmp_path / "template file.jinja") in " ".join(spec.serve.extra_args)
    assert "org/tokenizer" in spec.serve.extra_args


def test_inline_templates_uris_and_arbitrary_args_unchanged(tmp_path):
    raw = {"name": "m", "model_path": "s3://bucket/model", "serve_config": {"extra_args": {
        "--chat-template": "{{ messages[0]['content'] }}",
        "--generation-config": "auto", "--served-model-name": "org/model",
    }}}
    normalized = normalize_config(raw, tmp_path)
    assert normalized == raw


def test_explicit_dot_avoids_legacy_alias_and_expands_environment(monkeypatch, tmp_path):
    assert resolve_path("configs/model.yaml", tmp_path) == str(TOOL_ROOT / "configs/model.yaml")
    assert resolve_path("./configs/model.yaml", tmp_path) == str(tmp_path / "configs/model.yaml")
    monkeypatch.setenv("BATCHED_TEST_PATH_ROOT", str(tmp_path))
    assert resolve_path("$BATCHED_TEST_PATH_ROOT/a.yaml", TOOL_ROOT) == str(tmp_path / "a.yaml")
    assert resolve_path("~/key", tmp_path) == str(Path.home() / "key")


def test_ssh_keys_relative_to_cluster_config_and_missing_file_has_no_cwd_fallback(tmp_path):
    cfg = normalize_config({"ssh": {"hostname": "remote", "private_key": "./keys/id"}}, tmp_path)
    assert cfg["ssh"]["private_key"] == str(tmp_path / "keys/id")
    with pytest.raises(FileNotFoundError) as error:
        BenchmarkSpec.parse({"bench_param": "./missing.json"}, base_dir=tmp_path)
    assert str(tmp_path / "missing.json") in str(error.value)


def test_fingerprints_equal_across_cwd(monkeypatch, tmp_path):
    raw = yaml.safe_load((TOOL_ROOT / "configs/model.yaml").read_text())[0]
    values = []
    for cwd in (TOOL_ROOT, TOOL_ROOT.parents[1], tmp_path):
        monkeypatch.chdir(cwd)
        model = ModelSpec.parse(raw, base_dir=TOOL_ROOT / "configs")
        spec = plan_inference(model, text_case="configs/inference/text_case.yaml", image_case=None)
        values.append(InferenceTask(model, spec).fingerprint)
    assert len(set(values)) == 1


@pytest.mark.parametrize("entry", ["module", "script", "absolute"])
def test_script_entries_from_different_directories(tmp_path, entry):
    cwd, command = {
        "module": (TOOL_ROOT.parents[1], [sys.executable, "-m", "tools.batched_test.launch"]),
        "script": (TOOL_ROOT, [sys.executable, "launch.py"]),
        "absolute": (tmp_path, [sys.executable, str(TOOL_ROOT / "launch.py")]),
    }[entry]
    proc = subprocess.run(
        [*command, "--dry-run", "--gpus", "1"],
        cwd=cwd, capture_output=True, text=True, timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "GPU count filter: [1]" in proc.stdout
    assert "Selected " in proc.stdout
