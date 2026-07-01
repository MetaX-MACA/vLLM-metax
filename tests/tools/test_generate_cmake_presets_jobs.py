import json

import tools.generate_cmake_presets as presets


def test_generate_presets_accepts_parallelism_overrides(monkeypatch, tmp_path):
    output_path = tmp_path / "CMakeUserPresets.json"
    nvcc_path = tmp_path / "bin" / "nvcc"
    nvcc_path.parent.mkdir()
    nvcc_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(presets, "CUDA_HOME", str(tmp_path))
    monkeypatch.setattr(presets, "which", lambda name: None)
    monkeypatch.setattr(presets, "get_cpu_cores", lambda: 64)

    presets.generate_presets(
        output_path=str(output_path),
        force_overwrite=True,
        cmake_jobs=3,
        nvcc_threads=2,
    )

    data = json.loads(output_path.read_text())
    cache = data["configurePresets"][0]["cacheVariables"]
    assert cache["NVCC_THREADS"] == "2"
    assert data["buildPresets"][0]["jobs"] == 3


def test_generate_presets_handles_zero_cpu_count(monkeypatch, tmp_path):
    output_path = tmp_path / "CMakeUserPresets.json"
    nvcc_path = tmp_path / "bin" / "nvcc"
    nvcc_path.parent.mkdir()
    nvcc_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(presets, "CUDA_HOME", str(tmp_path))
    monkeypatch.setattr(presets, "which", lambda name: None)
    monkeypatch.setattr(presets, "get_cpu_cores", lambda: 0)

    presets.generate_presets(output_path=str(output_path), force_overwrite=True)

    data = json.loads(output_path.read_text())
    cache = data["configurePresets"][0]["cacheVariables"]
    assert cache["NVCC_THREADS"] == "1"
    assert data["buildPresets"][0]["jobs"] == 1
