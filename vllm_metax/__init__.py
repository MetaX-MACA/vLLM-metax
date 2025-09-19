# SPDX-License-Identifier: Apache-2.0

import importlib.metadata
import importlib.util
import os
import shutil
from pathlib import Path


def copy_with_backup(src_path: Path, dest_path: Path):
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
        backup_path = dest_full_path.parent / (dest_full_path.name + ".bak")
        if os.path.exists(backup_path):
            if os.path.isdir(backup_path) and not os.path.islink(backup_path):
                shutil.rmtree(backup_path)
            else:
                os.remove(backup_path)
        os.rename(dest_full_path, backup_path)

    # Perform the copy
    if os.path.isdir(src_path):
        shutil.copytree(src_path, dest_full_path)
    else:
        shutil.copy2(src_path, dest_full_path)


def post_installation():
    """Post installation script."""
    print("Post installation script.")

    # Get the path to the vllm distribution
    vllm_dist_path = Path(
        str(importlib.metadata.distribution("vllm").locate_file("vllm")))
    plugin_dist_path = Path(
        str(
            importlib.metadata.distribution("vllm_metax").locate_file(
                "vllm_metax")))

    assert (os.path.exists(vllm_dist_path))
    assert (os.path.exists(plugin_dist_path))

    print(f"vLLM Dist Location: [{vllm_dist_path}]")
    print(f"vLLM_plugin Dist Location: [{plugin_dist_path}]")

    files_to_copy = {
        "_C.abi3.so":
        vllm_dist_path,
        "_moe_C.abi3.so":
        vllm_dist_path,
        "cumem_allocator.abi3.so":
        vllm_dist_path,
        # TODO: workaround for torch 2.7 inferscheme, remove when torch >= 2.7
        "patch/vllm_substitution/fp8_utils.py":
        vllm_dist_path /
        "model_executor/layers/quantization/utils/fp8_utils.py",
        "patch/vllm_substitution/fused_moe.py":
        vllm_dist_path / "model_executor/layers/fused_moe/fused_moe.py",
    }

    for src_path, dest_path in files_to_copy.items():
        source_file = Path(plugin_dist_path) / src_path
        dest_file = Path(vllm_dist_path) / dest_path
        try:
            copy_with_backup(source_file, dest_file)
        except Exception as e:
            print("Init failed as: ", e)
            raise

    print("Post installation successful.")


def collect_env() -> None:
    from vllm_metax.collect_env import main as collect_env_main
    collect_env_main()


def register():
    """Register the METAX platform."""
    return "vllm_metax.platform.MacaPlatform"


def register_quant_configs():
    from vllm_metax.quant_config.awq import MacaAWQConfig  # noqa: F401
    from vllm_metax.quant_config.awq_marlin import (  # noqa: F401
        MacaAWQMarlinConfig)
    from vllm_metax.quant_config.gptq import MacaGPTQConfig  # noqa: F401
    from vllm_metax.quant_config.gptq_marlin import (  # noqa: F401
        MacaGPTQMarlinConfig)
    # HOTFIX: https://github.com/vllm-project/vllm/pull/22797
    from vllm_metax.quant_config.moe_wna16 import (  # noqa: F401
        MacaMoeWNA16Config)
    from vllm_metax.quant_config.compressed_tensors import (  # noqa: F401
        MacaCompressedTensorsConfig)


def register_model():
    import vllm_metax.patch  # noqa: F401

    register_quant_configs()

    from .models import register_model
    register_model()


def register_ops():
    import vllm_metax.ops  # noqa: F401
