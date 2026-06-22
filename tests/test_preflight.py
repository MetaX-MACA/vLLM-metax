import os
import stat
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vllm_metax.preflight import _run_version, collect_preflight


def _make_executable(path: Path) -> None:
    path.write_text("#!/bin/sh\necho fake-version\n", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def test_preflight_finds_sdk_local_tools(tmp_path):
    maca = tmp_path / "maca"
    cu_bridge = maca / "tools" / "cu-bridge"
    for directory in [
        maca / "bin",
        cu_bridge / "bin",
        cu_bridge / "CUDA_DIR" / "bin",
    ]:
        directory.mkdir(parents=True)
    (maca / "Version.txt").write_text("MACA Version: 2.33.1\n", encoding="utf-8")
    _make_executable(maca / "bin" / "mxcc")
    _make_executable(cu_bridge / "bin" / "cucc")
    _make_executable(cu_bridge / "CUDA_DIR" / "bin" / "nvcc")

    path = os.pathsep.join([str(maca / "bin"), os.environ.get("PATH", "")])
    with patch.dict(os.environ, {"MACA_PATH": str(maca), "PATH": path}, clear=True):
        info = collect_preflight()

    assert info["maca_version"] == "MACA Version: 2.33.1"
    assert info["commands"]["mxcc"].endswith("mxcc")
    assert info["commands"]["cucc"].endswith("cucc")
    assert info["commands"]["nvcc"].endswith("nvcc")


def test_run_version_returns_none_for_empty_output():
    with patch("subprocess.check_output", return_value=""):
        assert _run_version(["cucc", "--version"]) is None


def test_preflight_does_not_probe_current_directory_when_paths_are_missing():
    with patch.dict(os.environ, {}, clear=True):
        info = collect_preflight()

    assert info["commands"]["cucc"] is None
    assert info["commands"]["nvcc"] is None


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        test_preflight_finds_sdk_local_tools(Path(tmpdir))
