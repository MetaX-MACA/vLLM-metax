# SPDX-License-Identifier: Apache-2.0

import os
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    VLLM_TARGET_DEVICE: str = "cuda"
    MAX_JOBS: str | None
    NVCC_THREADS: str | None
    VLLM_USE_PRECOMPILED: bool = False
    VLLM_TEST_USE_PRECOMPILED_NIGHTLY_WHEEL: bool = False
    CMAKE_BUILD_TYPE: str | None
    VERBOSE: bool = False

environment_variables: dict[str, Callable[[], Any]] = {
    # ================== Installation Time Env Vars ==================
    # Target device of vLLM, supporting [cuda (by default),
    # rocm, neuron, cpu]
    "VLLM_TARGET_DEVICE": lambda: os.getenv("VLLM_TARGET_DEVICE", "cuda"),
    # Maximum number of compilation jobs to run in parallel.
    # By default this is the number of CPUs
    "MAX_JOBS": lambda: os.getenv("MAX_JOBS", None),
    # Number of threads to use for nvcc
    # By default this is 1.
    # If set, `MAX_JOBS` will be reduced to avoid oversubscribing the CPU.
    "NVCC_THREADS": lambda: os.getenv("NVCC_THREADS", None),
    # If set, vllm will use precompiled binaries (*.so)
    "VLLM_USE_PRECOMPILED": lambda: bool(os.environ.get("VLLM_USE_PRECOMPILED"))
    or bool(os.environ.get("VLLM_PRECOMPILED_WHEEL_LOCATION")),
    # CMake build type
    # If not set, defaults to "Debug" or "RelWithDebInfo"
    # Available options: "Debug", "Release", "RelWithDebInfo"
    "CMAKE_BUILD_TYPE": lambda: os.getenv("CMAKE_BUILD_TYPE"),
    # If set, vllm will print verbose logs during installation
    "VERBOSE": lambda: bool(int(os.getenv("VERBOSE", "0"))),
    # path to cudatoolkit home directory, under which should be bin, include,
    # and lib directories.
    "CUDA_HOME": lambda: os.environ.get("CUDA_HOME", None),
    # Path to the NCCL library file. It is needed because nccl>=2.19 brought
    # by PyTorch contains a bug: https://github.com/NVIDIA/nccl/issues/1234
    "VLLM_MCCL_SO_PATH": lambda: os.environ.get("VLLM_MCCL_SO_PATH", None),
    # when `VLLM_NCCL_SO_PATH` is not set, vllm will try to find the nccl
    # library file in the locations specified by `LD_LIBRARY_PATH`
    "LD_LIBRARY_PATH": lambda: os.environ.get("LD_LIBRARY_PATH", None),
    # ================== Runtime Env Vars ==================
    # When installing vllm from source, the version of vllm set by setuptool_scm
    # will be different from the version of vllm installed by pip.
    # (e.g. install vllm from source with tag v0.9.1 will cause the version set
    # as 0.9.2)
    "VLLM_OFFICIAL_VERSION": lambda: os.getenv("VLLM_OFFICIAL_VERSION", None),
}


# end-env-vars-definition
def override_vllm_env(env_name: str, value: Any, reason: str | None) -> None:
    """
    Override a vLLM environment variable at runtime.

    Args:
        env_key: environment variable name (e.g. "VLLM_USE_TRTLLM_ATTENTION")
                 or the callable from envs.environment_variables.
        value: new value. If None, the env var is removed from os.environ and
               the env resolver will return None.
    """
    from vllm import envs
    from vllm.logger import logger

    if not isinstance(env_name, str):
        raise TypeError("env_name must be a string")

    if env_name not in envs.environment_variables:
        raise KeyError(f"{env_name} is not a recognized vLLM environment variable")

    logger.info_once(
        "Note!: set %s to %s. Reason: %s",
        env_name,
        value,
        reason,
    )

    # Replace the resolver with a callable that returns the desired value.
    envs.environment_variables[env_name] = lambda v=value: v

    # Update os.environ for code that reads it directly.
    if value is None:
        os.environ.pop(env_name, None)
    else:
        if isinstance(value, bool):
            os.environ[env_name] = "1" if value else "0"
        else:
            os.environ[env_name] = str(value)


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())


def is_set(name: str):
    """Check if an environment variable is explicitly set."""
    if name in environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
