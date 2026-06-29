import pytest

import tools.generate_cmake_presets as presets


def test_no_prompt_fails_when_cuda_compiler_is_missing(monkeypatch):
    monkeypatch.setattr(presets, "CUDA_HOME", None)
    monkeypatch.setattr(presets, "which", lambda name: None)

    with pytest.raises(RuntimeError, match="--no-prompt"):
        presets.generate_presets(no_prompt=True)
