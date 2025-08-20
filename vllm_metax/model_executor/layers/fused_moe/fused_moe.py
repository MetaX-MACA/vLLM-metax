# SPDX-License-Identifier: Apache-2.0
"""Fused MoE kernel."""
import functools
import json
import os
from typing import Any, Dict, Optional

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.fused_moe import get_config_file_name

logger = init_logger(__name__)

@functools.lru_cache
def get_moe_configs(
    E: int,
    N: int,
    dtype: Optional[str],
    block_n: Optional[int] = None,
    block_k: Optional[int] = None,
# ┌------------------------  Metax Modification -------------------------┐
    H: int = 0,
# └------------------------- Metax Modification -------------------------┘
) -> Optional[dict[int, Any]]:
    """
    Return optimized configurations for the fused MoE kernel.

    The return value will be a dictionary that maps an irregular grid of
    batch sizes to configurations of the fused_moe kernel. To evaluate the
    kernel on a given batch size bs, the closest batch size in the grid should
    be picked and the associated configuration chosen to invoke the kernel.
    """

    # First look up if an optimized configuration is available in the configs
    # directory
    block_shape = [block_n, block_k] if block_n and block_k else None
    
# ┌------------------------  Metax Modification -------------------------┐
    json_file_name = get_config_file_name(E, N, dtype, block_shape)
    json_file_name_new = f"H={H},{json_file_name}"

    config_file_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name)
    config_file_path_new = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "configs", json_file_name_new)

    # First find H, E, N config file
    if os.path.exists(config_file_path_new):
        config_file_path = config_file_path_new

    if os.path.exists(config_file_path):
        with open(config_file_path) as f:
            logger.info("Using configuration from %s for MoE layer.",
                        config_file_path)
            # If a configuration has been found, return it
            return {int(key): val for key, val in json.load(f).items()}
# └------------------------- Metax Modification -------------------------┘
        
    # If no optimized configuration is available, we will use the default
    # configuration
    logger.warning(
        ("Using default MoE config. Performance might be sub-optimal! "
         "Config file not found at %s"), config_file_path)
    return None