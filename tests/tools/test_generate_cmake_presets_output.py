import json

import tools.generate_cmake_presets as presets


def test_generate_presets_accepts_custom_output(monkeypatch, tmp_path):
    output_path = tmp_path / "presets.json"
    nvcc_path = tmp_path / "bin" / "nvcc"
    nvcc_path.parent.mkdir()
    nvcc_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(presets, "CUDA_HOME", str(tmp_path))
    monkeypatch.setattr(presets, "which", lambda name: None)
    monkeypatch.setattr(presets, "get_cpu_cores", lambda: 8)

    presets.generate_presets(
        output_path=str(output_path),
        force_overwrite=True,
    )

    data = json.loads(output_path.read_text())
    compiler = data["configurePresets"][0]["cacheVariables"]["CMAKE_CUDA_COMPILER"]
    assert compiler == str(tmp_path / "bin" / "nvcc")
