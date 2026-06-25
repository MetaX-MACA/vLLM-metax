import os
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

import vllm_metax.collect_env as collect_env


def test_collect_env_reads_maca_home_when_maca_path_is_missing(tmp_path):
    (tmp_path / "Version.txt").write_text("MACA Version: 2.33.1\n", encoding="utf-8")

    with patch.dict(os.environ, {"MACA_HOME": str(tmp_path)}, clear=True):
        assert collect_env.get_maca_sdk_version(None).strip() == "2.33.1"


def test_collect_env_returns_none_without_maca_sdk_path():
    with patch.object(collect_env, "TORCH_AVAILABLE", False):
        with patch.dict(os.environ, {}, clear=True):
            assert collect_env.get_maca_sdk_version(None) is None
