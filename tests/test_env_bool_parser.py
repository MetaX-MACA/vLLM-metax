import os
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from vllm_metax.envs import _parse_bool_env


def test_bool_parser_accepts_zero_and_one():
    with patch.dict(os.environ, {"FLAG": "1"}, clear=True):
        assert _parse_bool_env("FLAG", "0") is True
    with patch.dict(os.environ, {"FLAG": "0"}, clear=True):
        assert _parse_bool_env("FLAG", "1") is False


def test_bool_parser_reports_variable_name():
    with patch.dict(os.environ, {"FLAG": "yes"}, clear=True):
        try:
            _parse_bool_env("FLAG", "0")
        except ValueError as err:
            assert "FLAG" in str(err)
            assert "0 or 1" in str(err)
        else:
            raise AssertionError("expected ValueError")


if __name__ == "__main__":
    test_bool_parser_accepts_zero_and_one()
    test_bool_parser_reports_variable_name()
