# SPDX-License-Identifier: Apache-2.0

from .version import __version__, __version_tuple__  # noqa: F401


def collect_env() -> None:
    from vllm_metax.collect_env import main as collect_env_main

    collect_env_main()


########### platform plugin ###########
def register():
    """Register the METAX platform."""
    return "vllm_metax.platform.MacaPlatform"


########### general plugins ###########
def register_patch():
    import vllm_metax.hotfix.fix_standalone_compile  # noqa: F401
    import vllm_metax.patch  # noqa: F401


def register_custom_op():
    import vllm_metax.customized  # noqa: F401


def register_quant_configs():
    import vllm_metax.quant_config  # noqa: F401


def register_customized():
    register_patch()
    register_custom_op()
    register_quant_configs()


def register_model():
    from .models import register_model

    register_model()
