# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import json

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import AutoTokenizer, LlamaTokenizerFast

from vllm_metax.patch.bugfix import tokenizer as maca_tokenizer


# Adapted from vLLM v0.33.0 tests/tokenizers_/test_detokenize.py.
@pytest.mark.parametrize("sample", [" hello", "  hello\nworld\t", "中  文\n"])
def test_llama_loader_preserves_serialized_bytelevel_backend(tmp_path, sample):
    backend = Tokenizer(models.BPE(unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        ["hello world", sample],
        trainers.BpeTrainer(
            vocab_size=275,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["<unk>", "<s>", "</s>"],
        ),
    )
    backend.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "LlamaTokenizerFast",
                "add_bos_token": False,
                "add_eos_token": False,
                "clean_up_tokenization_spaces": False,
            }
        )
    )

    assert (
        LlamaTokenizerFast.convert_to_native_format.__func__
        is maca_tokenizer.convert_to_native_format.__func__
    )
    tokenizer = AutoTokenizer.from_pretrained(tmp_path, trust_remote_code=False)
    expected_ids = backend.encode(sample, add_special_tokens=False).ids
    assert tokenizer.encode(sample, add_special_tokens=False) == expected_ids
    assert tokenizer.decode(expected_ids, skip_special_tokens=True) == sample
