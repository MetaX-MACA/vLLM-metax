import os
import unittest
from unittest.mock import patch

from vllm_metax.collect_env import (
    get_cuda_compiler_version_commands,
    get_running_cuda_version,
)


class TestCollectEnvCompilerVersion(unittest.TestCase):

    def test_prefers_maca_cucc_before_nvcc(self):
        with patch.dict(
            os.environ,
            {
                "CUCC_PATH": "/opt/maca/tools/cu-bridge",
                "MACA_PATH": "/opt/maca",
            },
            clear=True,
        ):
            commands = get_cuda_compiler_version_commands()

        self.assertEqual(
            commands[0],
            [
                os.path.join("/opt/maca/tools/cu-bridge", "bin", "cucc"),
                "--version",
            ],
        )
        self.assertIn(["cucc", "--version"], commands)
        self.assertLess(commands.index(["cucc", "--version"]), commands.index(["nvcc", "--version"]))
        self.assertEqual(commands[-1], ["nvcc", "--version"])

    def test_reads_first_available_compiler_version(self):
        calls = []

        def fake_run(command):
            calls.append(command)
            if command == ["cucc", "--version"]:
                return (
                    0,
                    "Cuda compilation tools, release 12.1, V12.1.105",
                    "",
                )
            return 127, "", "not found"

        with patch.dict(os.environ, {}, clear=True):
            version = get_running_cuda_version(fake_run)

        self.assertEqual(version, "12.1.105")
        self.assertIn(["cucc", "--version"], calls)
        self.assertNotIn(["nvcc", "--version"], calls)


if __name__ == "__main__":
    unittest.main()
