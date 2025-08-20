# SPDX-License-Identifier: Apache-2.0

from vllm_metax.utils import vllm_version
from packaging.version import parse

if parse(vllm_version()) >= parse("0.8.5"):
    from vllm_metax.patch.engine import patch_0_9_1
else:
    from vllm_metax.patch.engine import patch_0_8_5