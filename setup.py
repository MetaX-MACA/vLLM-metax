# SPDX-License-Identifier: Apache-2.0

import importlib.util
import logging
import os
import subprocess
import sys
from pathlib import Path
from shutil import which

import torch
from packaging.version import Version, parse
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from setuptools_scm import get_version
from setuptools_scm.version import ScmVersion
from torch.utils.cpp_extension import CUDA_HOME

try:
    from torch.utils.cpp_extension import MACA_HOME

    USE_MACA = True
except ImportError:
    MACA_HOME = None
    USE_MACA = False

CMAKE_EXECUTABLE = "cmake" if not USE_MACA else "cmake_maca"


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ROOT_DIR = Path(__file__).parent
logger = logging.getLogger(__name__)

# cannot import envs directly because it depends on vllm,
#  which is not installed yet
envs = load_module_from_path("envs", os.path.join(ROOT_DIR, "vllm_metax", "envs.py"))


VLLM_TARGET_DEVICE = envs.VLLM_TARGET_DEVICE
USE_PRECOMPILED_KERNEL = envs.USE_PRECOMPILED_KERNEL

if not (
    sys.platform.startswith("linux")
    or torch.version.cuda is None
    or os.getenv("VLLM_TARGET_DEVICE") != "cuda"
):
    # if cuda is not available and VLLM_TARGET_DEVICE is not set, abort
    raise AssertionError("Plugin only support cuda on Linux platform.")

MAIN_CUDA_VERSION = "12.8"


def is_sccache_available() -> bool:
    return which("sccache") is not None


def is_ccache_available() -> bool:
    return which("ccache") is not None


def is_ninja_available() -> bool:
    return which("ninja") is not None


def is_url_available(url: str) -> bool:
    from urllib.request import urlopen

    status = None
    try:
        with urlopen(url) as f:
            status = f.status
    except Exception:
        return False
    return status == 200


class CMakeExtension(Extension):
    def __init__(self, name: str, cmake_lists_dir: str = ".", **kwa) -> None:
        super().__init__(name, sources=[], py_limited_api=True, **kwa)
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class cmake_build_ext(build_ext):
    # A dict of extension directories that have been configured.
    did_config: dict[str, bool] = {}

    #
    # Determine number of compilation jobs and optionally nvcc compile threads.
    #
    def compute_num_jobs(self):
        # `num_jobs` is either the value of the MAX_JOBS environment variable
        # (if defined) or the number of CPUs available.
        num_jobs = envs.MAX_JOBS
        if num_jobs is not None:
            num_jobs = int(num_jobs)
            logger.info("Using MAX_JOBS=%d as the number of jobs.", num_jobs)
        else:
            try:
                # os.sched_getaffinity() isn't universally available, so fall
                #  back to os.cpu_count() if we get an error here.
                num_jobs = len(os.sched_getaffinity(0))
            except AttributeError:
                num_jobs = os.cpu_count()

        nvcc_threads = 1

        return num_jobs, nvcc_threads

    #
    # Perform cmake configuration for a single extension.
    #
    def configure(self, ext: CMakeExtension) -> None:
        # If we've already configured using the CMakeLists.txt for
        # this extension, exit early.
        if ext.cmake_lists_dir in cmake_build_ext.did_config:
            return

        cmake_build_ext.did_config[ext.cmake_lists_dir] = True

        # Select the build type.
        # Note: optimization level + debug info are set by the build type
        default_cfg = "Debug" if self.debug else "RelWithDebInfo"
        cfg = envs.CMAKE_BUILD_TYPE or default_cfg

        cmake_args = [
            "-DCMAKE_BUILD_TYPE={}".format(cfg),
            "-DVLLM_TARGET_DEVICE={}".format(VLLM_TARGET_DEVICE),
        ]

        verbose = envs.VERBOSE
        if verbose:
            cmake_args += ["-DCMAKE_VERBOSE_MAKEFILE=ON"]

        if is_sccache_available():
            cmake_args += [
                "-DCMAKE_C_COMPILER_LAUNCHER=sccache",
                "-DCMAKE_CXX_COMPILER_LAUNCHER=sccache",
                "-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache",
                "-DCMAKE_HIP_COMPILER_LAUNCHER=sccache",
            ]
        elif is_ccache_available():
            cmake_args += [
                "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_HIP_COMPILER_LAUNCHER=ccache",
            ]

        # Pass the python executable to cmake so it can find an exact
        # match.
        cmake_args += ["-DVLLM_PYTHON_EXECUTABLE={}".format(sys.executable)]

        # Pass the python path to cmake so it can reuse the build dependencies
        # on subsequent calls to python.
        cmake_args += ["-DVLLM_PYTHON_PATH={}".format(":".join(sys.path))]

        # Override the base directory for FetchContent downloads to $ROOT/.deps
        # This allows sharing dependencies between profiles,
        # and plays more nicely with sccache.
        # To override this, set the FETCHCONTENT_BASE_DIR environment variable.
        fc_base_dir = os.path.join(ROOT_DIR, ".deps")
        fc_base_dir = os.environ.get("FETCHCONTENT_BASE_DIR", fc_base_dir)
        cmake_args += ["-DFETCHCONTENT_BASE_DIR={}".format(fc_base_dir)]

        #
        # Setup parallelism and build tool
        #
        num_jobs, nvcc_threads = self.compute_num_jobs()

        if nvcc_threads:
            cmake_args += ["-DNVCC_THREADS={}".format(nvcc_threads)]

        if is_ninja_available():
            build_tool = ["-G", "Ninja"]
            cmake_args += [
                "-DCMAKE_JOB_POOL_COMPILE:STRING=compile",
                "-DCMAKE_JOB_POOLS:STRING=compile={}".format(num_jobs),
            ]
        else:
            # Default build tool to whatever cmake picks.
            build_tool = []

        # Make sure we use the nvcc from CUDA_HOME
        if _is_maca():
            cmake_args += [f"-DCMAKE_CUDA_COMPILER={CUDA_HOME}/bin/nvcc"]
            cmake_args += ["-DUSE_MACA=1"]

        if not _build_custom_ops():
            cmake_args += ["-DUSE_PRECOMPILED_KERNEL=ON"]
        subprocess.check_call(
            [CMAKE_EXECUTABLE, ext.cmake_lists_dir, *build_tool, *cmake_args],
            cwd=self.build_temp,
        )

    def build_extensions(self) -> None:
        # Ensure that CMake is present and working
        try:
            subprocess.check_output([CMAKE_EXECUTABLE, "--version"])
        except OSError as e:
            raise RuntimeError("Cannot find cmake_maca executable") from e

        # Create build directory if it does not exist.
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        targets = []

        def target_name(s: str) -> str:
            return s.removeprefix("vllm_metax.").removeprefix("vllm_flash_attn.")

        # Build all the extensions
        for ext in self.extensions:
            self.configure(ext)
            targets.append(target_name(ext.name))

        num_jobs, _ = self.compute_num_jobs()

        build_args = [
            "--build",
            ".",
            f"-j={num_jobs}",
            *[f"--target={name}" for name in targets],
        ]

        subprocess.check_call([CMAKE_EXECUTABLE, *build_args], cwd=self.build_temp)

        # Install the libraries
        for ext in self.extensions:
            # Install the extension into the proper location
            outdir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()

            # Skip if the install directory is the same as the build directory
            if outdir == self.build_temp:
                continue

            # CMake appends the extension prefix to the install path,
            # and outdir already contains that prefix, so we need to remove it.
            prefix = outdir
            for _ in range(ext.name.count(".")):
                prefix = prefix.parent

            # prefix here should actually be the same for all components
            install_args = [
                CMAKE_EXECUTABLE,
                "--install",
                ".",
                "--prefix",
                prefix,
                "--component",
                target_name(ext.name),
            ]
            subprocess.check_call(install_args, cwd=self.build_temp)

    def run(self):
        # First, run the standard build_ext command to compile the extensions
        super().run()


def _is_maca() -> bool:
    has_cuda = torch.version.cuda is not None
    return VLLM_TARGET_DEVICE == "cuda" and has_cuda


def _build_custom_ops() -> bool:
    return _is_maca() and not envs.USE_PRECOMPILED_KERNEL


def get_maca_version() -> Version:
    """
    Returns the MACA SDK Version
    """
    file_full_path = os.path.join(os.getenv("MACA_PATH"), "Version.txt")
    if not os.path.isfile(file_full_path):
        return None

    with open(file_full_path, encoding="utf-8") as file:
        first_line = file.readline().strip()
    return parse(first_line.split(":")[-1])


def fixed_version_scheme(version: ScmVersion) -> str:
    return "0.13.0"


def always_hash(version: ScmVersion) -> str:
    """
    Always include short commit hash and current date (YYYYMMDD)
    """
    from datetime import datetime

    date_str = datetime.now().strftime("%Y%m%d")
    if version.node is not None:
        short_hash = version.node[:7]  # short commit id
        return f"{short_hash}.d{date_str}"
    return f"unknown.{date_str}"


def get_plugin_version() -> str:
    version = get_version(
        version_scheme=fixed_version_scheme,
        local_scheme=always_hash,
        write_to="vllm_metax/_version.py",
    )
    sep = "+" if "+" not in version else "."  # dev versions might contain +

    if _is_maca():
        maca_version_str = str(get_maca_version())
        torch_version = torch.__version__
        major_minor_version = ".".join(torch_version.split(".")[:2])
        version += f"{sep}maca{maca_version_str}.torch{major_minor_version}"
    else:
        raise RuntimeError("Unknown runtime environment")

    return version


def get_requirements() -> list[str]:
    """Get Python package dependencies from requirements.txt."""
    requirements_dir = ROOT_DIR / "requirements"

    def _read_requirements(filename: str) -> list[str]:
        with open(requirements_dir / filename) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            elif (
                not line.startswith("--")
                and not line.startswith("#")
                and line.strip() != ""
            ):
                resolved_requirements.append(line)
        return resolved_requirements

    if _is_maca():
        requirements = _read_requirements("maca.txt")
        modified_requirements = []
        for req in requirements:
            modified_requirements.append(req)
        requirements = modified_requirements
    else:
        raise ValueError("Unsupported platform, please use CUDA.")
    return requirements


ext_modules = []


ext_modules.append(CMakeExtension(name="vllm_metax.cumem_allocator"))

if _build_custom_ops():
    ext_modules.append(CMakeExtension(name="vllm_metax._C"))
    ext_modules.append(CMakeExtension(name="vllm_metax._moe_C"))
else:
    print("Using precompiled kernels, skipping building custom ops.")


package_data = {
    "vllm_metax": [
        "py.typed",
        "model_executor/layers/fused_moe/configs/*.json",
    ]
}


class custom_install(install):
    def run(self):
        install.run(self)


cmdclass = (
    {
        "build_ext": cmake_build_ext,
    }
    if ext_modules
    else {}
)

setup(
    # static metadata should rather go in pyproject.toml
    version=get_plugin_version(),
    ext_modules=ext_modules,
    install_requires=get_requirements(),
    extras_require={
        "bench": ["pandas", "datasets"],
        "tensorizer": ["tensorizer>=2.9.0"],
        "fastsafetensors": ["fastsafetensors >= 0.1.10"],
        "runai": ["runai-model-streamer", "runai-model-streamer-s3", "boto3"],
        "audio": ["librosa", "soundfile"],  # Required for audio processing
        "video": [],  # Kept for backwards compatibility
    },
    cmdclass=cmdclass,
    package_data=package_data,
)
