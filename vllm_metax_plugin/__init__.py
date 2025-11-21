# 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import shutil
import importlib.metadata
import importlib.util
from pathlib import Path

def copy_with_backup(src, dst):
    src = Path(src)
    dst = Path(dst)
    if dst.exists():
        dst_backup = dst.with_suffix(dst.suffix + '.bak')
        dst.rename(dst_backup)
    shutil.copy2(src, dst)
    print(f"Copied {src} to {dst}")


def post_installation():
    """Post installation script."""
    print("Post installation script.")

    # Get the path to the vllm distribution
    vllm_dist_path = importlib.metadata.distribution(
        "vllm").locate_file("vllm")
    plugin_dist_path = importlib.metadata.distribution(
        "vllm_metax_plugin").locate_file("vllm_metax_plugin")

    assert (os.path.exists(vllm_dist_path))
    assert (os.path.exists(plugin_dist_path))

    print(f"vLLM Dist Location: [{vllm_dist_path}]")
    print(f"vLLM_plugin Dist Location: [{plugin_dist_path}]")

    files_to_copy = [
        "_C.abi3.so",
        "_moe_C.abi3.so",
        "cumem_allocator.abi3.so",
    ]

    for file_name in files_to_copy:
        source_file = Path(plugin_dist_path) / file_name
        dest_file = Path(vllm_dist_path) / file_name
        try:
            copy_with_backup(source_file, dest_file)
        except importlib.metadata.PackageNotFoundError:
            print("vllm not installed - breakup installation")
            raise

    print("Post installation successful.")


def register():
    """Register the METAX platform."""
    return "vllm_metax_plugin.platform.MetaXPlatform"


def register_model():
    import vllm_metax_plugin.patch
    from .models import register_model
    register_model()
