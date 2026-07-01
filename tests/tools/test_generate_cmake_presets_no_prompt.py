import pytest

import tools.generate_cmake_presets as presets


def test_no_prompt_fails_when_cuda_compiler_is_missing(monkeypatch):
    monkeypatch.setattr(presets, "CUDA_HOME", None)
    monkeypatch.setattr(presets, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="--no-prompt"):
        presets.generate_presets(no_prompt=True)


def test_no_prompt_fails_when_output_exists(monkeypatch, tmp_path):
    output = tmp_path / "CMakeUserPresets.json"
    output.write_text("{}", encoding="utf-8")
    nvcc_path = tmp_path / "bin" / "nvcc"
    nvcc_path.parent.mkdir()
    nvcc_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(presets, "CUDA_HOME", str(tmp_path))
    monkeypatch.setattr(presets, "which", lambda name: None)
    monkeypatch.setattr(presets, "get_cpu_cores", lambda: 8)

    with pytest.raises(RuntimeError, match="force-overwrite"):
        presets.generate_presets(output_path=str(output), no_prompt=True)
