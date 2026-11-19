# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import json

import pytest

import vllm_metax


@pytest.mark.requires_maca
def test_public_engine_entrypoint_applies_maca_plugin_config(tmp_path):
    from vllm.engine.arg_utils import EngineArgs
    from vllm.platforms import current_platform

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["OPTForCausalLM"],
                "model_type": "opt",
                "hidden_size": 32,
                "ffn_dim": 64,
                "num_hidden_layers": 2,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "vocab_size": 64,
                "max_position_embeddings": 64,
                "word_embed_proj_dim": 32,
                "do_layer_norm_before": True,
                "layer_norm_eps": 1e-5,
                "activation_function": "relu",
                "torch_dtype": "bfloat16",
                "pad_token_id": 1,
                "bos_token_id": 2,
                "eos_token_id": 2,
                "init_std": 0.02,
            }
        )
    )
    config = EngineArgs(
        model=str(tmp_path),
        tokenizer=str(tmp_path),
        skip_tokenizer_init=True,
        load_format="dummy",
        dtype="bfloat16",
        max_model_len=64,
        enforce_eager=True,
    ).create_engine_config()

    assert current_platform.is_maca()
    assert config.model_config.architectures == ["OPTForCausalLM"]
    assert config.model_config.dtype == __import__("torch").bfloat16
    assert config.model_config.sleep_mode_backend == "macamem"
    assert config.parallel_config.worker_cls
