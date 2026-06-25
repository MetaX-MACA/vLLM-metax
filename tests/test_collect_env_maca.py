import os
import unittest
from unittest.mock import patch

from vllm_metax.collect_env import get_env_vars


class TestCollectEnvMacaVars(unittest.TestCase):

    def test_collects_maca_runtime_variables(self):
        env = {
            "MACA_PATH": "/opt/maca",
            "CUCC_PATH": "/opt/maca/tools/cu-bridge",
            "MCCL_DEBUG": "INFO",
            "MX_VISIBLE_DEVICES": "0",
            "GITLINK_TOKEN": "must-not-leak",
            "OPENAI_API_KEY": "must-not-leak-either",
        }
        with patch.dict(os.environ, env, clear=True):
            output = get_env_vars()

        self.assertIn("MACA_PATH=/opt/maca", output)
        self.assertIn("CUCC_PATH=/opt/maca/tools/cu-bridge", output)
        self.assertIn("MCCL_DEBUG=INFO", output)
        self.assertIn("MX_VISIBLE_DEVICES=0", output)
        self.assertNotIn("GITLINK_TOKEN", output)
        self.assertNotIn("OPENAI_API_KEY", output)
        self.assertNotIn("must-not-leak", output)

    def test_does_not_duplicate_known_prefixed_variables(self):
        with patch.dict(
            os.environ,
            {"MACA_VLLM_ENABLE_MCTLASS_PYTHON_API": "1"},
            clear=True,
        ):
            lines = get_env_vars().splitlines()

        self.assertEqual(lines.count("MACA_VLLM_ENABLE_MCTLASS_PYTHON_API=1"), 1)


if __name__ == "__main__":
    unittest.main()
