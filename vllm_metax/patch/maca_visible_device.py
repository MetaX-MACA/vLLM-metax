# SPDX-License-Identifier: Apache-2.0
# 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved. 
from vllm.worker.worker_base import WorkerWrapperBase, logger
import contextlib

from typing import List, Dict, Iterator
import os
from vllm.utils import update_environment_variables
from vllm.platforms import current_platform
from vllm.v1.engine.utils import get_device_indices
from unittest.mock import patch
from vllm.config import VllmConfig


def update_environment_variables_with_maca(
        self, envs_list: List[Dict[str, str]]) -> None:
    envs = envs_list[self.rpc_rank]
    key = 'CUDA_VISIBLE_DEVICES'
    # sync `MACA_VISIBLE_DEVICES`` with `CUDA_VISIBLE_DEVICES`
    envs['MACA_VISIBLE_DEVICES'] = envs.get(key, '')
    if key in envs and key in os.environ:
        # overwriting CUDA_VISIBLE_DEVICES is desired behavior
        # suppress the warning in `update_environment_variables`
        del os.environ[key]
    update_environment_variables(envs)


@contextlib.contextmanager
def set_device_control_env_var_with_maca(vllm_config: VllmConfig,
                                         local_dp_rank: int) -> Iterator[None]:
    """
    Temporarily set CUDA_VISIBLE_DEVICES or equivalent
    for engine subprocess.
    """
    world_size = vllm_config.parallel_config.world_size
    evar = current_platform.device_control_env_var

    value = get_device_indices(evar, local_dp_rank, world_size)
    with patch.dict(os.environ, values=((evar, value), )):
        os.environ['MACA_VISIBLE_DEVICES'] = value
        yield


from vllm.v1.engine import utils

utils.set_device_control_env_var = set_device_control_env_var_with_maca
WorkerWrapperBase.update_environment_variables = update_environment_variables_with_maca
