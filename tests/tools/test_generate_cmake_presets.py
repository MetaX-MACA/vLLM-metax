import json

from tools.generate_cmake_presets import generate_presets


def test_generate_presets_accepts_explicit_cuda_compiler(tmp_path):
    output = tmp_path / "CMakeUserPresets.json"

    generate_presets(
        output_path=str(output),
        force_overwrite=True,
        cuda_compiler="/opt/maca/bin/cucc",
    )

    data = json.loads(output.read_text(encoding="utf-8"))
    cache = data["configurePresets"][0]["cacheVariables"]
    assert cache["CMAKE_CUDA_COMPILER"] == "/opt/maca/bin/cucc"
