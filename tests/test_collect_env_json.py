import json
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from vllm_metax.collect_env import SystemEnv, get_json_env_info


def test_collect_env_json_uses_namedtuple_fields():
    dummy_data = {field: f"mock_{field}" for field in SystemEnv._fields}
    mock_env = SystemEnv(**dummy_data)

    with patch("vllm_metax.collect_env.get_env_info", return_value=mock_env):
        data = json.loads(get_json_env_info())

    assert data == dummy_data


if __name__ == "__main__":
    test_collect_env_json_uses_namedtuple_fields()
