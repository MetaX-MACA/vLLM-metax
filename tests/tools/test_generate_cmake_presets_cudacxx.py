import json

import tools.generate_cmake_presets as presets


def test_generate_presets_prefers_cudacxx(monkeypatch, tmp_path):
    output_path = tmp_path / "CMakeUserPresets.json"
    cudacxx = tmp_path / "cu-bridge" / "bin" / "nvcc"

    monkeypatch.setenv("CUDACXX", str(cudacxx))
    monkeypatch.setattr(presets, "CUDA_HOME", str(tmp_path / "cuda"))
    monkeypatch.setattr(presets.os.path, "exists", lambda path: True)
    monkeypatch.setattr(presets, "which", lambda name: None)
    monkeypatch.setattr(presets, "get_cpu_cores", lambda: 8)

    presets.generate_presets(output_path=str(output_path), force_overwrite=True)

    data = json.loads(output_path.read_text())
    compiler = data["configurePresets"][0]["cacheVariables"]["CMAKE_CUDA_COMPILER"]
    assert compiler == str(cudacxx)
