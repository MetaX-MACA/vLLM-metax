import json

import tools.generate_cmake_presets as presets


def test_generate_presets_prefers_cudacxx(monkeypatch, tmp_path):
    output_path = tmp_path / "CMakeUserPresets.json"
    cudacxx = tmp_path / "cu-bridge" / "bin" / "nvcc"
    cudacxx.parent.mkdir(parents=True)
    cudacxx.write_text("", encoding="utf-8")

    monkeypatch.setenv("CUDACXX", str(cudacxx))
    monkeypatch.setattr(presets, "CUDA_HOME", str(tmp_path / "cuda"))
    monkeypatch.setattr(presets, "which", lambda name: None)
    monkeypatch.setattr(presets, "get_cpu_cores", lambda: 8)

    presets.generate_presets(output_path=str(output_path), force_overwrite=True)

    data = json.loads(output_path.read_text())
    compiler = data["configurePresets"][0]["cacheVariables"]["CMAKE_CUDA_COMPILER"]
    assert compiler == str(cudacxx)


def test_generate_presets_resolves_cudacxx_from_path(monkeypatch, tmp_path):
    output_path = tmp_path / "CMakeUserPresets.json"
    cudacxx = tmp_path / "bin" / "cucc"
    cudacxx.parent.mkdir()
    cudacxx.write_text("", encoding="utf-8")

    monkeypatch.setenv("CUDACXX", "cucc")
    monkeypatch.setattr(presets, "CUDA_HOME", None)
    monkeypatch.setattr(
        presets,
        "which",
        lambda name: str(cudacxx) if name == "cucc" else None,
    )
    monkeypatch.setattr(presets, "get_cpu_cores", lambda: 8)

    presets.generate_presets(output_path=str(output_path), force_overwrite=True)

    data = json.loads(output_path.read_text())
    compiler = data["configurePresets"][0]["cacheVariables"]["CMAKE_CUDA_COMPILER"]
    assert compiler == str(cudacxx)
