# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from setuptools import setup

setup(
    name="colbert-query-plugin",
    version="0.1",
    packages=["colbert_query_processor"],
    entry_points={
        "vllm.io_processor_plugins": [
            "colbert_query_plugin = colbert_query_processor:register_colbert_query_embedding_processor",  # noqa: E501
        ]
    },
)
