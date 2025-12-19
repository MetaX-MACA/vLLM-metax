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


def register_ops():
    register_patch()
    import vllm_metax.ops  # noqa: F401


def register_model():
    from .models import register_model

    register_model()


def register_quant_configs():
    from vllm_metax.quant_config.awq import MacaAWQConfig  # noqa: F401
    from vllm_metax.quant_config.awq_marlin import (  # noqa: F401
        MacaAWQMarlinConfig,
    )
    from vllm_metax.quant_config.gptq import MacaGPTQConfig  # noqa: F401
    from vllm_metax.quant_config.gptq_marlin import (  # noqa: F401
        MacaGPTQMarlinConfig,
    )
    from vllm_metax.quant_config.moe_wna16 import (  # noqa: F401
        MacaMoeWNA16Config,
    )
    from vllm_metax.quant_config.compressed_tensors import (  # noqa: F401
        MacaCompressedTensorsConfig,
    )
