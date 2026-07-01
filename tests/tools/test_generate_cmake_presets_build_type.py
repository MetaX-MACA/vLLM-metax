import json

import tools.generate_cmake_presets as presets


def test_generate_presets_uses_requested_build_type(monkeypatch, tmp_path):
    output_path = tmp_path / "CMakeUserPresets.json"
    nvcc_path = tmp_path / "bin" / "nvcc"
    nvcc_path.parent.mkdir()
    nvcc_path.write_text("", encoding="utf-8")

    monkeypatch.setattr(presets, "CUDA_HOME", str(tmp_path))
    monkeypatch.setattr(presets, "which", lambda name: None)
    monkeypatch.setattr(presets, "get_cpu_cores", lambda: 8)

    presets.generate_presets(
        output_path=str(output_path),
        force_overwrite=True,
        build_type="RelWithDebInfo",
    )

    data = json.loads(output_path.read_text())
    configure = data["configurePresets"][0]
    assert configure["name"] == "relwithdebinfo"
    assert configure["binaryDir"] == "${sourceDir}/cmake-build-relwithdebinfo"
    assert configure["cacheVariables"]["CMAKE_BUILD_TYPE"] == "RelWithDebInfo"
    assert data["buildPresets"][0]["configurePreset"] == "relwithdebinfo"
