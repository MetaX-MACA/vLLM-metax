# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

from vllm.model_executor.layers.quantization.utils import quant_utils


def maca_get_attribute_fallback(obj, attributes: list[str]):
    for attr in attributes:
        # ┌------------------------  Metax Modification -------------------------┐
        if hasattr(obj, attr) and getattr(obj, attr) is not None:
            return getattr(obj, attr)
        # └------------------------- Metax Modification -------------------------┘
    raise AttributeError(f"'{obj}' has no recognized attributes: {attributes}.")


quant_utils.get_attribute_fallback = maca_get_attribute_fallback
