import pytest

import tools.generate_cmake_presets as presets


def test_generate_presets_rejects_missing_cuda_compiler(monkeypatch):
    monkeypatch.setattr(presets, "CUDA_HOME", None)
    monkeypatch.setattr(presets, "which", lambda name: None)
    monkeypatch.setattr("builtins.input", lambda prompt: "/missing/nvcc")

    with pytest.raises(FileNotFoundError, match="/missing/nvcc"):
        presets.generate_presets(force_overwrite=True)


def test_generate_presets_rejects_compiler_directory(monkeypatch, tmp_path):
    compiler_dir = tmp_path / "bin"
    compiler_dir.mkdir()

    monkeypatch.setattr(presets, "CUDA_HOME", None)
    monkeypatch.setattr(presets, "which", lambda name: None)
    monkeypatch.setattr("builtins.input", lambda prompt: str(compiler_dir))

    with pytest.raises(FileNotFoundError, match="CUDA compiler not found"):
        presets.generate_presets(force_overwrite=True)
