# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# -----------------------------------------------------------------------------
# Note: Provide a pyNCCL-compatible MCCL wrapper for MetaX distributed
#       communication.
#
# Affected versions: vLLM 0.29.1.dev0 (98dff2a81d), verified 2026-09-17.
#
# Remove at: Upstream distributed communication supports MCCL library symbols without
#     replacing the NCCL bindings.
# -----------------------------------------------------------------------------


from vllm_metax.patch.utils import patch
from vllm_metax.distributed.device_communicators.pynccl_wrapper import MCCLLibrary

patch("vllm.distributed.device_communicators.pynccl", "NCCLLibrary")(MCCLLibrary)
patch("vllm.distributed.device_communicators.pynccl_wrapper", "NCCLLibrary")(
    MCCLLibrary
)
