# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import logging
import os
import re
import shutil
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

CMAKE_EXECUTABLE = 'cmake' if not USE_MACA else 'cmake_maca'


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
envs = load_module_from_path('envs',
                             os.path.join(ROOT_DIR, 'vllm_metax', 'envs.py'))

try:
    vllm_dist_path = importlib.metadata.distribution("vllm").locate_file(
        "vllm")
    logger.info("detected vllm distribution path: %s", vllm_dist_path)
except importlib.metadata.PackageNotFoundError:
    vllm_dist_path = None
    logger.warning("vllm not installed! You need to install vllm first. ")
except Exception:
    vllm_dist_path = None
    logger.warning("Error getting vllm distribution path")

VLLM_TARGET_DEVICE = envs.VLLM_TARGET_DEVICE

if not (sys.platform.startswith("linux") or torch.version.cuda is None
        or os.getenv("VLLM_TARGET_DEVICE") != "cuda"):
    # if cuda or hip is not available and VLLM_TARGET_DEVICE is not set,
    # fallback to cpu
    raise AssertionError("Plugin only support cuda on linux platform. ")

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

    def __init__(self, name: str, cmake_lists_dir: str = '.', **kwa) -> None:
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
            '-DCMAKE_BUILD_TYPE={}'.format(cfg),
            '-DVLLM_TARGET_DEVICE={}'.format(VLLM_TARGET_DEVICE),
        ]

        verbose = envs.VERBOSE
        if verbose:
            cmake_args += ['-DCMAKE_VERBOSE_MAKEFILE=ON']

        if is_sccache_available():
            cmake_args += [
                '-DCMAKE_C_COMPILER_LAUNCHER=sccache',
                '-DCMAKE_CXX_COMPILER_LAUNCHER=sccache',
                '-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache',
                '-DCMAKE_HIP_COMPILER_LAUNCHER=sccache',
            ]
        elif is_ccache_available():
            cmake_args += [
                '-DCMAKE_C_COMPILER_LAUNCHER=ccache',
                '-DCMAKE_CXX_COMPILER_LAUNCHER=ccache',
                '-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache',
                '-DCMAKE_HIP_COMPILER_LAUNCHER=ccache',
            ]

        # Pass the python executable to cmake so it can find an exact
        # match.
        cmake_args += ['-DVLLM_PYTHON_EXECUTABLE={}'.format(sys.executable)]

        # Pass the python path to cmake so it can reuse the build dependencies
        # on subsequent calls to python.
        cmake_args += ['-DVLLM_PYTHON_PATH={}'.format(":".join(sys.path))]

        # Override the base directory for FetchContent downloads to $ROOT/.deps
        # This allows sharing dependencies between profiles,
        # and plays more nicely with sccache.
        # To override this, set the FETCHCONTENT_BASE_DIR environment variable.
        fc_base_dir = os.path.join(ROOT_DIR, ".deps")
        fc_base_dir = os.environ.get("FETCHCONTENT_BASE_DIR", fc_base_dir)
        cmake_args += ['-DFETCHCONTENT_BASE_DIR={}'.format(fc_base_dir)]

        #
        # Setup parallelism and build tool
        #
        num_jobs, nvcc_threads = self.compute_num_jobs()

        if nvcc_threads:
            cmake_args += ['-DNVCC_THREADS={}'.format(nvcc_threads)]

        if is_ninja_available():
            build_tool = ['-G', 'Ninja']
            cmake_args += [
                '-DCMAKE_JOB_POOL_COMPILE:STRING=compile',
                '-DCMAKE_JOB_POOLS:STRING=compile={}'.format(num_jobs),
            ]
        else:
            # Default build tool to whatever cmake picks.
            build_tool = []

        # Make sure we use the nvcc from CUDA_HOME
        if _is_cuda() and not USE_MACA:
            cmake_args += [f'-DCMAKE_CUDA_COMPILER={CUDA_HOME}/bin/nvcc']
        if USE_MACA:
            cmake_args += ['-DUSE_MACA=1']
        subprocess.check_call(
            [CMAKE_EXECUTABLE, ext.cmake_lists_dir, *build_tool, *cmake_args],
            cwd=self.build_temp)

    def build_extensions(self) -> None:
        # Ensure that CMake is present and working
        try:
            subprocess.check_output([CMAKE_EXECUTABLE, '--version'])
        except OSError as e:
            raise RuntimeError('Cannot find CMake executable') from e

        # Create build directory if it does not exist.
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        targets = []

        def target_name(s: str) -> str:
            return s.removeprefix("vllm_metax.").removeprefix(
                "vllm_flash_attn.")

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

        subprocess.check_call([CMAKE_EXECUTABLE, *build_args],
                              cwd=self.build_temp)

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
            for _ in range(ext.name.count('.')):
                prefix = prefix.parent

            # prefix here should actually be the same for all components
            install_args = [
                CMAKE_EXECUTABLE, "--install", ".", "--prefix", prefix,
                "--component",
                target_name(ext.name)
            ]
            subprocess.check_call(install_args, cwd=self.build_temp)

    def run(self):
        # First, run the standard build_ext command to compile the extensions
        super().run()


class repackage_wheel(build_ext):
    """Extracts libraries and other files from an existing wheel."""

    def get_base_commit_in_main_branch(self) -> str:
        # Force to use the nightly wheel. This is mainly used for CI testing.
        if envs.VLLM_TEST_USE_PRECOMPILED_NIGHTLY_WHEEL:
            return "nightly"

        try:
            # Get the latest commit hash of the upstream main branch.
            resp_json = subprocess.check_output([
                "curl", "-s",
                "https://api.github.com/repos/vllm-project/vllm/commits/main"
            ]).decode("utf-8")
            upstream_main_commit = json.loads(resp_json)["sha"]

            # Check if the upstream_main_commit exists in the local repo
            try:
                subprocess.check_output(
                    ["git", "cat-file", "-e", f"{upstream_main_commit}"])
            except subprocess.CalledProcessError:
                # If not present, fetch it from the remote repository.
                # Note that this does not update any local branches,
                # but ensures that this commit ref and its history are
                # available in our local repo.
                subprocess.check_call([
                    "git", "fetch", "https://github.com/vllm-project/vllm",
                    "main"
                ])

            # Then get the commit hash of the current branch that is the same as
            # the upstream main commit.
            current_branch = subprocess.check_output(
                ["git", "branch", "--show-current"]).decode("utf-8").strip()

            base_commit = subprocess.check_output([
                "git", "merge-base", f"{upstream_main_commit}", current_branch
            ]).decode("utf-8").strip()
            return base_commit
        except ValueError as err:
            raise ValueError(err) from None
        except Exception as err:
            logger.warning(
                "Failed to get the base commit in the main branch. "
                "Using the nightly wheel. The libraries in this "
                "wheel may not be compatible with your dev branch: %s", err)
            return "nightly"

    def run(self) -> None:
        assert _is_cuda(
        ), "VLLM_USE_PRECOMPILED is only supported for CUDA builds"

        wheel_location = os.getenv("VLLM_PRECOMPILED_WHEEL_LOCATION", None)
        if wheel_location is None:
            base_commit = self.get_base_commit_in_main_branch()
            wheel_location = f"https://wheels.vllm.ai/{base_commit}/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl"
            # Fallback to nightly wheel if latest commit wheel is unavailable,
            # in this rare case, the nightly release CI hasn't finished on main.
            if not is_url_available(wheel_location):
                wheel_location = "https://wheels.vllm.ai/nightly/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl"

        import zipfile

        if os.path.isfile(wheel_location):
            wheel_path = wheel_location
            print(f"Using existing wheel={wheel_path}")
        else:
            # Download the wheel from a given URL, assume
            # the filename is the last part of the URL
            wheel_filename = wheel_location.split("/")[-1]

            import tempfile

            # create a temporary directory to store the wheel
            temp_dir = tempfile.mkdtemp(prefix="vllm-wheels")
            wheel_path = os.path.join(temp_dir, wheel_filename)

            print(f"Downloading wheel from {wheel_location} to {wheel_path}")

            from urllib.request import urlretrieve

            try:
                urlretrieve(wheel_location, filename=wheel_path)
            except Exception as e:
                from setuptools.errors import SetupError

                raise SetupError(
                    f"Failed to get vLLM wheel from {wheel_location}") from e

        with zipfile.ZipFile(wheel_path) as wheel:
            files_to_copy = [
                "vllm_metax/_C.abi3.so",
                "vllm_metax/_moe_C.abi3.so",
                "vllm_metax/cumem_allocator.abi3.so",
                # "vllm/_version.py", # not available in nightly wheels yet
            ]

            file_members = list(
                filter(lambda x: x.filename in files_to_copy, wheel.filelist))

            # vllm_flash_attn python code:
            # Regex from
            #  `glob.translate('vllm/vllm_flash_attn/**/*.py', recursive=True)`
            compiled_regex = re.compile(
                r"vllm/vllm_flash_attn/(?:[^/.][^/]*/)*(?!\.)[^/]*\.py")
            file_members += list(
                filter(lambda x: compiled_regex.match(x.filename),
                       wheel.filelist))

            for file in file_members:
                print(f"Extracting and including {file.filename} "
                      "from existing wheel")
                package_name = os.path.dirname(file.filename).replace("/", ".")
                file_name = os.path.basename(file.filename)

                if package_name not in package_data:
                    package_data[package_name] = []

                wheel.extract(file)
                if file_name.endswith(".py"):
                    # python files shouldn't be added to package_data
                    continue

                package_data[package_name].append(file_name)


def _is_cuda() -> bool:
    has_cuda = torch.version.cuda is not None
    return (VLLM_TARGET_DEVICE == "cuda" and has_cuda)


def _build_custom_ops() -> bool:
    return _is_cuda()


def get_maca_version() -> Version:
    """
    Returns the MACA SDK Version
    """
    file_full_path = os.path.join(os.getenv('MACA_PATH'), 'Version.txt')
    if not os.path.isfile(file_full_path):
        return None

    with open(file_full_path, encoding='utf-8') as file:
        first_line = file.readline().strip()
    return parse(first_line.split(":")[-1])


def fixed_version_scheme(version: ScmVersion) -> str:
    return "0.11.0"


def always_hash(version: ScmVersion) -> str:
    """
    Always include short commit hash and current date (YYYYMMDD)
    """
    from datetime import datetime
    date_str = datetime.now().strftime("%Y%m%d")
    if version.node is not None:
        short_hash = version.node[:7]  # short commit id
        return f"g{short_hash}.d{date_str}"
    return f"unknown.{date_str}"


def get_vllm_version() -> str:
    version = get_version(version_scheme=fixed_version_scheme,
                          local_scheme=always_hash,
                          write_to="vllm_metax/_version.py")
    sep = "+" if "+" not in version else "."  # dev versions might contain +

    if _is_cuda():
        if envs.VLLM_USE_PRECOMPILED:
            version += f"{sep}precompiled"
        else:
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
            elif not line.startswith("--") and not line.startswith(
                    "#") and line.strip() != "":
                resolved_requirements.append(line)
        return resolved_requirements

    if _is_cuda():
        requirements = _read_requirements("maca.txt")
        cuda_major, cuda_minor = torch.version.cuda.split(".")
        modified_requirements = []
        for req in requirements:
            if ("vllm-flash-attn" in req and cuda_major != "12"):
                # vllm-flash-attn is built only for CUDA 12.x.
                # Skip for other versions.
                continue
            modified_requirements.append(req)
        requirements = modified_requirements
    else:
        raise ValueError(
            "Unsupported platform, please use CUDA, ROCm, Neuron, HPU, "
            "or CPU.")
    return requirements


ext_modules = []

if _is_cuda():
    ext_modules.append(CMakeExtension(name="vllm_metax._moe_C"))
    ext_modules.append(CMakeExtension(name="vllm_metax.cumem_allocator"))

if _build_custom_ops() or True:
    ext_modules.append(CMakeExtension(name="vllm_metax._C"))

package_data = {
    "vllm_metax": [
        "py.typed",
        "model_executor/layers/fused_moe/configs/*.json",
        "model_executor/layers/quantization/utils/configs/*.json",
        "attention/backends/configs/*.json",
    ]
}


class custom_install(install):

    def _copy_with_backup(self, src_path: Path, dest_path: Path):
        """
        Copy a file or directory from src_path to dest_path.
        - If dest_path is an existing directory, copy src_path into that directory.
        - If dest_path exists as a file or directory, back it up as .bak before copying.
        """
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Source path does not exist: {src_path}")

        # If dest_path is an existing directory, copy into it
        if os.path.isdir(dest_path):
            dest_full_path = dest_path / os.path.basename(src_path)
        else:
            dest_full_path = dest_path

        # Backup if target path already exists (file or dir)
        if os.path.exists(dest_full_path):
            backup_path = dest_full_path.parent / (dest_full_path.name +
                                                   ".bak")
            logger.debug(
                f"{dest_full_path} exists, backing it up to {backup_path}")
            if os.path.exists(backup_path):
                logger.debug(
                    f"Backup path {backup_path} already exists, removing it.")
                if os.path.isdir(
                        backup_path) and not os.path.islink(backup_path):
                    shutil.rmtree(backup_path)
                else:
                    os.remove(backup_path)
            os.rename(dest_full_path, backup_path)

        # Perform the copy
        if os.path.isdir(src_path):
            shutil.copytree(src_path, dest_full_path)
        else:
            shutil.copy2(src_path, dest_full_path)

        logger.info(f"Copied {src_path} to {dest_full_path}")

    def _copy_files_to_vllm(self, src_path: Path, dest_path: Path):
        try:
            self._copy_with_backup(src_path, dest_path)
        except Exception as e:
            logger.error(f"Error copying files: {e}")

    def run(self):
        install.run(self)

        if not vllm_dist_path:
            return

        files_to_copy = {
            # for get_available_device: set cuda
            "vllm_metax/patch/vllm_substitution/utils.py":
            vllm_dist_path / "model_executor/layers/fla/ops/utils.py",
        }

        for src_path, dest_path in files_to_copy.items():
            source_file = Path(self.build_lib) / src_path
            self._copy_files_to_vllm(source_file, dest_path)


if not ext_modules:
    cmdclass = {}
else:
    cmdclass = {
        "build_ext":
        repackage_wheel if envs.VLLM_USE_PRECOMPILED else cmake_build_ext,
        "install": custom_install
    }

setup(
    # static metadata should rather go in pyproject.toml
    version=get_vllm_version(),
    ext_modules=ext_modules,
    install_requires=get_requirements(),
    extras_require={
        "bench": ["pandas", "datasets"],
        "tensorizer": ["tensorizer>=2.9.0"],
        "fastsafetensors": ["fastsafetensors >= 0.1.10"],
        "runai": ["runai-model-streamer", "runai-model-streamer-s3", "boto3"],
        "audio": ["librosa", "soundfile"],  # Required for audio processing
        "video": []  # Kept for backwards compatibility
    },
    cmdclass=cmdclass,
    package_data=package_data,
)
