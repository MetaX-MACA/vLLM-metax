# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

from .version import __version__, __version_tuple__  # noqa: F401


def collect_env() -> None:
    from vllm_metax.collect_env import main as collect_env_main

    collect_env_main()


########### platform plugin ###########
def register():
    """Register the METAX platform."""
    return "vllm_metax.platform.MacaPlatform"


########### general plugins ###########
def _patch():
    import vllm_metax.patch.bugfix  # noqa: F401
    import vllm_metax.patch.enhancement  # noqa: F401
    import vllm_metax.patch.performance  # noqa: F401


def _out_of_tree():
    import vllm_metax.registry  # noqa: F401


def register_out_of_tree():
    _patch()
    _out_of_tree()


def register_model():
    from .models import register_model

    register_model()
