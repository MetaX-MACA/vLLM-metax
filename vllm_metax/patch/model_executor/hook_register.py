# SPDX-License-Identifier: Apache-2.0

from vllm.model_executor.layers.quantization import (
    _CUSTOMIZED_METHOD_TO_QUANT_CONFIG, QUANTIZATION_METHODS,
    QuantizationConfig, register_quantization_config)


def register_quantization_config(quantization: str):
    """Register a customized vllm quantization config.

    When a quantization method is not supported by vllm, you can register a customized
    quantization config to support it.

    Args:
        quantization (str): The quantization method name.

    Examples:
        >>> from vllm.model_executor.layers.quantization import register_quantization_config
        >>> from vllm.model_executor.layers.quantization import get_quantization_config
        >>> from vllm.model_executor.layers.quantization.base_config import QuantizationConfig
        >>>
        >>> @register_quantization_config("my_quant")
        ... class MyQuantConfig(QuantizationConfig):
        ...     pass
        >>>
        >>> get_quantization_config("my_quant")
        <class 'MyQuantConfig'>
    """  # noqa: E501

    def _wrapper(quant_config_cls):
        if not issubclass(quant_config_cls, QuantizationConfig):
            raise ValueError("The quantization config must be a subclass of "
                             "`QuantizationConfig`.")
        _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization] = quant_config_cls
        QUANTIZATION_METHODS.append(quantization)
        return quant_config_cls

    return _wrapper


import vllm.model_executor.layers.quantization

vllm.model_executor.layers.quantization.register_quantization_config = register_quantization_config
