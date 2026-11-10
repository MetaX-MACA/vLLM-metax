# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract tests; vendor calls are mocked, so these prove no GPU numerics.

Run in isolation from model fixtures:
pytest --confcutdir=tests/v1/attention tests/v1/attention/test_metax_fa_contract.py
"""

from types import SimpleNamespace
from unittest.mock import create_autospec

import pytest
import torch
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import AttentionType

from vllm_metax.third_party import fa_interface as api
from vllm_metax.v1.attention.backends import fa_utils as utils
from vllm_metax.v1.attention.backends.flash_attn import (
    MacaFlashAttentionBackend as Backend,
)
from vllm_metax.v1.attention.backends.flash_attn_diffkv import (
    FlashAttentionDiffKVBackend,
)


@pytest.fixture(autouse=True)
def available_packages(monkeypatch):
    monkeypatch.setattr(api, "FA2_AVAILABLE", True)
    monkeypatch.setattr(api, "FA3_AVAILABLE", True)
    monkeypatch.setattr(utils, "get_current_vllm_config_or_none", lambda: None)
    monkeypatch.setattr(
        utils,
        "current_platform",
        SimpleNamespace(
            get_device_capability=lambda: DeviceCapability(8, 9),
        ),
    )


@pytest.mark.parametrize(
    "version,fp8,sizes",
    [
        (2, False, {32, 64, 96, 128, 160, 192, 256, 512}),
        (2, True, set()),
        (3, False, {64, 128, 192, 256}),
        (3, True, {64, 96, 128, 192, 256}),
    ],
)
def test_equal_dimensions(version, fp8, sizes):
    for size in range(0, 545, 8):
        assert api.is_fa_geometry_supported(version, size, is_fp8=fp8) == (
            size in sizes
        )


@pytest.mark.parametrize("version,fp8", [(2, False), (3, False), (3, True)])
@pytest.mark.parametrize(
    "qk,v,expected",
    [(192, 128, True), (128, 192, False), (176, 128, False), (192, 112, False)],
)
def test_diff_dimensions(version, fp8, qk, v, expected):
    assert api.is_fa_geometry_supported(version, qk, v, is_fp8=fp8) is expected


@pytest.mark.parametrize(
    "block,expected",
    [
        (None, True),
        (0, False),
        (-2, False),
        (1, True),
        (2, True),
        (8, True),
        (16, True),
        (256, True),
        (512, True),
        (1024, True),
        (3, False),
        (6, False),
        (24, False),
        (384, False),
    ],
)
def test_block_sizes(block, expected):
    assert Backend.supports_block_size(block) is expected
    assert api.is_fa_geometry_supported(3, 128, block_size=block) is expected
    if block is not None and expected:
        assert Backend.get_supported_kernel_block_sizes(
            SimpleNamespace(block_size=block)
        ) == [block]


@pytest.mark.parametrize(
    "size,kv,expected",
    [
        (32, "auto", 2),
        (96, "auto", 2),
        (160, "bfloat16", 2),
        (512, "auto", 2),
        (64, "auto", 2),
        (128, "bfloat16", 2),
        (192, "auto", 2),
        (256, "auto", 2),
        (96, "fp8", 3),
        (96, "fp8_e4m3", 3),
        (160, "fp8", None),
        (32, "fp8", None),
        (512, "fp8", None),
        (80, "auto", None),
        (128, "fp8_e5m2", None),
        (128, "fp8_per_token_head", None),
        (128, "int8_per_token_head", None),
    ],
)
def test_selection_and_dtype_predicate(size, kv, expected):
    assert utils.get_flash_attn_version(head_size=size, kv_cache_dtype=kv) == expected
    assert utils.flash_attn_supports_kv_cache_dtype(kv, head_size=size) is (
        expected is not None
    )


@pytest.mark.parametrize(
    "feature",
    [
        dict(requires_alibi=True),
        dict(has_sinks=True),
    ],
)
def test_fallback_preserves_features(feature):
    assert utils.get_flash_attn_version(head_size=128, **feature) == 2
    assert (
        utils.get_flash_attn_version(head_size=128, kv_cache_dtype="fp8", **feature)
        is None
    )


@pytest.mark.parametrize(
    "cap",
    [
        None,
        DeviceCapability(7, 5),
        DeviceCapability(8, 0),
        DeviceCapability(8, 6),
        DeviceCapability(8, 9),
        DeviceCapability(9, 0),
        DeviceCapability(10, 0),
    ],
)
@pytest.mark.parametrize("requested", [None, 2, 3])
@pytest.mark.parametrize("kv", ["auto", "float16", "bfloat16", "fp8", "fp8_e4m3"])
def test_hardware_selection(monkeypatch, cap, requested, kv):
    monkeypatch.setattr(
        utils, "current_platform", SimpleNamespace(get_device_capability=lambda: cap)
    )
    monkeypatch.setattr(
        utils,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(
            attention_config=SimpleNamespace(flash_attn_version=requested)
        ),
    )
    expected = 3 if cap == DeviceCapability(8, 9) and requested == 3 else 2
    if kv.startswith("fp8") and cap == DeviceCapability(8, 9) and requested != 2:
        expected = 3
    if cap is None or cap < DeviceCapability(8, 0):
        expected = None
    if kv.startswith("fp8") and (cap != DeviceCapability(8, 9) or requested == 2):
        expected = None
    assert utils.get_flash_attn_version(head_size=128, kv_cache_dtype=kv) == expected
    assert utils.flash_attn_supports_kv_cache_dtype(kv, head_size=128) is (
        expected is not None
    )
    if cap is not None:
        assert Backend.supports_compute_capability(cap) is (
            cap >= DeviceCapability(8, 0)
        )


@pytest.mark.parametrize("requested", [None, 2, 3])
@pytest.mark.parametrize(
    "cap",
    [
        DeviceCapability(8, 0),
        DeviceCapability(8, 6),
        DeviceCapability(8, 9),
        DeviceCapability(9, 0),
        DeviceCapability(10, 0),
    ],
)
def test_non_fp8_supports_all_maca_devices(monkeypatch, requested, cap):
    def unexpected_query():
        pytest.fail("The supplied device capability must take precedence")

    monkeypatch.setattr(
        utils,
        "current_platform",
        SimpleNamespace(get_device_capability=unexpected_query),
    )
    monkeypatch.setattr(
        utils,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(
            attention_config=SimpleNamespace(flash_attn_version=requested)
        ),
    )
    assert utils.get_flash_attn_version(head_size=128, device_capability=cap) == (
        3 if requested == 3 and cap == DeviceCapability(8, 9) else 2
    )


@pytest.mark.parametrize(
    "cap",
    [
        DeviceCapability(7, 5),
        DeviceCapability(8, 0),
        DeviceCapability(8, 9),
        DeviceCapability(9, 0),
        DeviceCapability(10, 0),
    ],
)
@pytest.mark.parametrize("kv", ["auto", "fp8"])
def test_backend_combination_uses_supplied_capability(monkeypatch, cap, kv):
    monkeypatch.setattr(
        utils,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(attention_config=SimpleNamespace(flash_attn_version=3)),
    )
    # The fixture's platform reports 8.9; the supplied capability must take precedence.
    reason = Backend.supports_combination(
        128, torch.bfloat16, kv, 16, False, False, False, False, cap
    )
    assert (reason is None) is (
        cap >= DeviceCapability(8, 0)
        and (kv == "auto" or cap == DeviceCapability(8, 9))
    )


@pytest.mark.parametrize(
    "cap",
    [
        DeviceCapability(8, 0),
        DeviceCapability(8, 6),
        DeviceCapability(8, 9),
        DeviceCapability(9, 0),
    ],
)
def test_non_fp8_fa3_feature_fallback_on_maca_devices(monkeypatch, cap):
    monkeypatch.setattr(
        utils, "current_platform", SimpleNamespace(get_device_capability=lambda: cap)
    )
    monkeypatch.setattr(
        utils,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(attention_config=SimpleNamespace(flash_attn_version=3)),
    )
    assert utils.get_flash_attn_version(head_size=96) == 2
    assert utils.get_flash_attn_version(head_size=128, requires_alibi=True) == 2


@pytest.mark.parametrize(
    "requested,kv,expected",
    [
        (2, "auto", 2),
        (2, "fp8", None),
        (3, "auto", 3),
        (3, "fp8", 3),
        (4, "auto", None),
    ],
)
def test_explicit_version(monkeypatch, requested, kv, expected):
    monkeypatch.setattr(
        utils,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(
            attention_config=SimpleNamespace(flash_attn_version=requested)
        ),
    )
    assert utils.get_flash_attn_version(head_size=128, kv_cache_dtype=kv) == expected


def test_missing_packages(monkeypatch):
    monkeypatch.setattr(api, "FA2_AVAILABLE", False)
    assert utils.is_flash_attn_varlen_func_available()
    assert Backend.supports_head_size(128) is True
    assert Backend.supports_head_size(32) is False
    assert utils.get_flash_attn_version(head_size=96) is None
    assert FlashAttentionDiffKVBackend.is_supported_on_current_device(192, 128, False)
    assert not FlashAttentionDiffKVBackend.is_supported_on_current_device(
        192, 128, True
    )
    monkeypatch.setattr(api, "FA3_AVAILABLE", False)
    assert not utils.is_flash_attn_varlen_func_available()
    assert utils.get_flash_attn_version() is None
    assert not api.is_fa_version_supported(None)
    assert not api.is_fa_version_supported(99)


@pytest.mark.parametrize("fa2,fa3", [(True, False), (False, True), (False, False)])
def test_mla_prefill_availability(monkeypatch, fa2, fa3):
    from vllm_metax.v1.attention.backends.mla.prefill.flash_attn import (
        MacaFlashAttnPrefillBackend,
    )

    monkeypatch.setattr(api, "FA2_AVAILABLE", fa2)
    monkeypatch.setattr(api, "FA3_AVAILABLE", fa3)
    assert MacaFlashAttnPrefillBackend.is_available() is (fa2 or fa3)


@pytest.mark.parametrize("version", [2, 3])
@pytest.mark.parametrize("qk_dim,v_dim", [(192, 128), (256, 256), (128, 128)])
@pytest.mark.parametrize("context_chunk", [False, True])
def test_mla_prefill_varlen_dispatch(monkeypatch, version, qk_dim, v_dim, context_chunk):
    from vllm_metax.v1.attention.backends.mla.prefill.flash_attn import (
        MacaFlashAttnPrefillBackend,
    )

    config = SimpleNamespace(attention_config=SimpleNamespace(flash_attn_version=version))
    monkeypatch.setattr(utils, "get_current_vllm_config_or_none", lambda: config)
    backend = MacaFlashAttnPrefillBackend(
        num_heads=4,
        scale=0.1,
        kv_lora_rank=512,
        qk_nope_head_dim=qk_dim - 64,
        qk_rope_head_dim=64,
        v_head_dim=v_dim,
        vllm_config=config,
    )
    assert backend.vllm_flash_attn_version == version
    assert backend.supports_out()
    starts = torch.tensor([0, 1, 4], dtype=torch.int32)
    backend.prepare_metadata(SimpleNamespace(query_start_loc=starts, max_query_len=3))
    q = torch.zeros(4, 4, qk_dim, dtype=torch.bfloat16)
    v = torch.zeros(4, 4, v_dim, dtype=torch.bfloat16)
    result = torch.ones_like(v)
    lse = torch.zeros(4, 4)
    package = api.flash_attn if version == 2 else api.flash_attn_3
    vendor = create_autospec(package.flash_attn_varlen_func, return_value=(result, lse))
    cache = create_autospec(package.flash_attn_with_kvcache)
    monkeypatch.setattr(package, "flash_attn_varlen_func", vendor)
    monkeypatch.setattr(package, "flash_attn_with_kvcache", cache)
    out = torch.empty_like(v)
    if context_chunk:
        answer = backend.run_prefill_context_chunk(
            SimpleNamespace(
                query_start_loc=starts, cu_seq_lens=starts, max_query_len=3, max_seq_len=3
            ),
            q=q, k=q, v=v, out=out,
        )
    else:
        answer = backend.run_prefill_new_tokens(
            q=q, k=q, v=v, return_softmax_lse=True, out=out
        )
    assert answer[0] is out and answer[1] is lse
    torch.testing.assert_close(out, result)
    assert vendor.call_args.kwargs["causal"] is (not context_chunk)
    vendor.assert_called_once()
    cache.assert_not_called()


def varlen_inputs(paged=True, fp8=False, diffkv=False):
    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    dim = 192 if diffkv else (96 if fp8 else 128)
    dim_v = 128 if diffkv else dim
    return dict(
        q=torch.empty(4, 4, dim, dtype=dtype),
        k=torch.empty((4, 4, 2, dim) if paged else (10, 2, dim), dtype=dtype),
        v=torch.empty((4, 4, 2, dim_v) if paged else (10, 2, dim_v), dtype=dtype),
        cu_seqlens_q=torch.tensor([0, 1, 4], dtype=torch.int32),
        max_seqlen_q=3,
        seqused_k=torch.tensor([3, 7], dtype=torch.int32),
        max_seqlen_k=7,
        block_table=torch.tensor([[2, 0], [1, 3]], dtype=torch.int32)
        if paged
        else None,
    )


@pytest.mark.parametrize(
    "version,paged,fp8,diffkv",
    [
        (2, False, False, False),
        (2, True, False, True),
        (3, False, False, False),
        (3, False, True, False),
    ],
)
@pytest.mark.parametrize("lengths", ["used", "cumulative"])
@pytest.mark.parametrize("lse_flag", [None, "return_softmax_lse", "return_attn_probs"])
def test_varlen_dispatch(monkeypatch, version, paged, fp8, diffkv, lengths, lse_flag):
    args = varlen_inputs(paged, fp8, diffkv)
    if lengths == "cumulative":
        args.pop("seqused_k")
        args["cu_seqlens_k"] = torch.tensor([0, 3, 10], dtype=torch.int32)
    package = api.flash_attn if version == 2 else api.flash_attn_3
    name = "flash_attn_varlen_func"
    # Bind to the real installed signature so incorrect keyword names fail.
    mock = create_autospec(getattr(package, name))
    result = torch.zeros(4, 4, args["v"].shape[-1], dtype=torch.bfloat16)
    lse = torch.zeros(4, 4, dtype=torch.float32)
    mock.return_value = (result, lse, None) if lse_flag else result
    monkeypatch.setattr(package, name, mock)
    out = torch.empty_like(result)
    answer = api.flash_attn_varlen_func(
        **args, out=out, fa_version=version, **({lse_flag: True} if lse_flag else {})
    )
    assert (answer[0] if lse_flag else answer) is out
    torch.testing.assert_close(out, result)
    if lse_flag:
        assert len(answer) == 2 and answer[1] is lse
    kw = mock.call_args.kwargs
    assert kw["softmax_scale"] is None  # Preserve vendor scaling, including q_v.
    torch.testing.assert_close(
        kw["cu_seqlens_k"], torch.tensor([0, 3, 10], dtype=torch.int32)
    )
    assert kw["return_attn_probs"] is bool(lse_flag)


@pytest.mark.parametrize(
    "name", ["q_descale", "k_descale", "v_descale", "scheduler_metadata", "q_v"]
)
@pytest.mark.parametrize("cache", [False, True])
def test_fa2_ignores_each_unsupported_argument(monkeypatch, name, cache):
    args = varlen_inputs()
    api_name = "flash_attn_with_kvcache" if cache else "flash_attn_varlen_func"
    result = torch.zeros_like(args["q"])
    if cache:
        args = dict(
            q=args["q"].reshape(2, 2, 4, 128), k_cache=args["k"], v_cache=args["v"]
        )
        result = result.reshape(2, 2, 4, 128)
    vendor = create_autospec(getattr(api.flash_attn, api_name), return_value=result)
    monkeypatch.setattr(api.flash_attn, api_name, vendor)
    answer = getattr(api, api_name)(**args, **{name: torch.ones(1)}, fa_version=2)
    assert answer is result
    assert name not in vendor.call_args.kwargs


@pytest.mark.parametrize("version", [2, 3])
@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize(
    "name,value",
    [
        ("output_scale", torch.ones(1)),
        ("mask_mod", object()),
        ("block_sparse_tensors", object()),
        ("aux_tensors", [torch.ones(1)]),
        ("aux_tensor_leading_dims", []),
        ("dynamic_causal", torch.tensor([True, False])),
        ("cp_world_size", 2),
        ("cp_rank", 1),
        ("cp_tot_seqused_k", torch.ones(2)),
        ("gather_kv_indices", torch.ones(4, 128, dtype=torch.int32)),
        ("future_optional_argument", object()),
    ],
)
def test_unified_optional_arguments_are_ignored(
    monkeypatch, version, cache, name, value
):
    args = varlen_inputs(paged=cache)
    result = torch.zeros_like(args["q"])
    if cache:
        args = dict(
            q=args["q"].reshape(2, 2, 4, 128), k_cache=args["k"], v_cache=args["v"]
        )
        result = result.reshape(2, 2, 4, 128)
    api_name = "flash_attn_with_kvcache" if cache else "flash_attn_varlen_func"
    package = api.flash_attn if version == 2 else api.flash_attn_3
    vendor = create_autospec(getattr(package, api_name), return_value=result)
    monkeypatch.setattr(package, api_name, vendor)
    assert getattr(api, api_name)(**args, fa_version=version, **{name: value}) is result
    assert name not in vendor.call_args.kwargs


@pytest.mark.parametrize("version", [2, 3])
def test_invalid_page_rejected_before_kernel(version):
    args = varlen_inputs()
    k = torch.empty(4, 6, 2, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="geometry"):
        api.flash_attn_with_kvcache(
            q=args["q"].reshape(2, 2, 4, 128),
            k_cache=k,
            v_cache=torch.empty_like(k),
            block_table=args["block_table"],
            fa_version=version,
        )


@pytest.mark.parametrize("version", [2, 3])
def test_cache_dispatch(monkeypatch, version):
    args = varlen_inputs()
    package = api.flash_attn if version == 2 else api.flash_attn_3
    mock = create_autospec(package.flash_attn_with_kvcache)
    q = args["q"].reshape(2, 2, 4, 128)
    out = torch.empty_like(q)
    lse = torch.zeros(2, 4, 2)
    mock.return_value = (
        (torch.zeros_like(q), lse) if version == 3 else torch.zeros_like(q)
    )
    monkeypatch.setattr(package, "flash_attn_with_kvcache", mock)
    result = api.flash_attn_with_kvcache(
        q=q,
        k_cache=args["k"],
        v_cache=args["v"],
        block_table=args["block_table"],
        cache_seqlens=args["seqused_k"],
        return_softmax_lse=version == 3,
        out=out,
        fa_version=version,
    )
    assert (result[0] if version == 3 else result) is out
    kw = mock.call_args.kwargs
    assert kw["page_table" if version == 3 else "block_table"] is args["block_table"]


@pytest.mark.parametrize("diffkv", [False, True])
@pytest.mark.parametrize("fp8", [False, True])
def test_layer_dispatch_preserves_variable_query_lengths(monkeypatch, diffkv, fp8):
    from vllm_metax.v1.attention.backends import flash_attn as regular
    from vllm_metax.v1.attention.backends.flash_attn_diffkv import (
        FlashAttentionDiffKVImpl,
    )

    monkeypatch.setattr(
        utils,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(attention_config=SimpleNamespace(flash_attn_version=3)),
    )
    monkeypatch.setenv("VLLM_METAX_ENABLE_FA_SPLIT_FORWARD", "0")
    cls = FlashAttentionDiffKVImpl if diffkv else regular.FlashAttentionImpl
    # Skip distributed group discovery; initialize the real attention implementation.
    impl = object.__new__(cls)
    impl.dcp_world_size = 1
    impl.__init__(
        num_heads=4,
        head_size=192 if diffkv else 128,
        scale=0.1,
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="fp8" if fp8 else "auto",
    )
    assert impl.vllm_flash_attn_version == 3
    assert impl.head_size_v == 128 if diffkv else impl.head_size_v is None
    args = varlen_inputs(paged=True, fp8=False, diffkv=diffkv)
    if fp8:
        args["q"] = torch.empty_like(args["q"], dtype=torch.float8_e4m3fn)
    cache_dtype = torch.uint8 if fp8 else torch.bfloat16
    cache = torch.empty(4, 2, 4, impl.head_size + 128, dtype=cache_dtype)
    metadata = SimpleNamespace(
        num_actual_tokens=4,
        query_start_loc=args["cu_seqlens_q"],
        seq_lens=args["seqused_k"],
        max_query_len=3,
        max_seq_len=7,
        block_table=args["block_table"],
        causal=True,
        scheduler_metadata=None,
        mm_prefix_query_range_tensor=None,
        rswa_prefix_lens=None,
    )
    out = torch.empty(4, 4, 128, dtype=torch.bfloat16)
    vendor = create_autospec(
        api.flash_attn_3.flash_attn_with_kvcache, return_value=torch.zeros_like(out)
    )
    monkeypatch.setattr(api.flash_attn_3, "flash_attn_with_kvcache", vendor)
    layer = SimpleNamespace(
        _q_scale=torch.tensor(0.5),
        _k_scale=torch.tensor(0.25),
        _v_scale=torch.tensor(0.125),
    )
    assert impl.forward(layer, args["q"], None, None, cache, metadata, out) is out
    kw = vendor.call_args.kwargs
    assert kw["q"].shape[0] == 4
    assert kw["cu_seqlens_q"] is args["cu_seqlens_q"]
    assert kw["cache_seqlens"] is args["seqused_k"]
    assert kw["q_descale"].shape == (2, 2)
    assert kw["k_descale"][0, 0] == 0.25
    assert kw["v_descale"][0, 0] == 0.125


@pytest.mark.parametrize(
    "size,kv,block,expected",
    [
        (96, "auto", 16, None),
        (96, "fp8", 16, None),
        (96, "auto", 24, "No MetaX"),
        (160, "fp8", 16, "No MetaX"),
        (80, "auto", 16, "No MetaX"),
    ],
)
def test_backend_combination_matches_selector(size, kv, block, expected):
    reason = Backend.supports_combination(
        size,
        torch.bfloat16,
        kv,
        block,
        False,
        False,
        False,
        False,
        DeviceCapability(8, 9),
    )
    assert reason is None if expected is None else expected in reason


def test_diffkv_never_rounds_unsupported_dimensions():
    assert not FlashAttentionDiffKVBackend.is_supported_on_current_device(
        176, 128, False
    )
    assert not FlashAttentionDiffKVBackend.is_supported_on_current_device(
        192, 112, False
    )
    assert FlashAttentionDiffKVBackend.is_supported_on_current_device(192, 128, False)


def test_default_kernel_sizes_are_exact_powers_of_two():
    sizes = Backend.get_supported_kernel_block_sizes()
    assert sizes and 1 in sizes and 256 in sizes
    assert all(
        isinstance(size, int) and size > 0 and size & (size - 1) == 0 for size in sizes
    )


def test_fp8_dcp_rejected_before_dispatch(monkeypatch):
    from vllm_metax.v1.attention.backends import flash_attn as regular

    monkeypatch.setattr(
        utils,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(attention_config=SimpleNamespace(flash_attn_version=3)),
    )
    monkeypatch.setattr(
        regular,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(
            parallel_config=SimpleNamespace(decode_context_parallel_size=2)
        ),
    )
    impl = object.__new__(regular.FlashAttentionImpl)
    with pytest.raises(NotImplementedError, match="FP8 KV with DCP"):
        impl.__init__(
            num_heads=4,
            head_size=128,
            scale=0.1,
            num_kv_heads=2,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype="fp8",
        )


@pytest.mark.parametrize(
    "fp8,diffkv", [(False, False), (True, False), (False, True), (True, True)]
)
def test_fa3_varlen_ignores_block_table_without_redirecting(monkeypatch, fp8, diffkv):
    args = varlen_inputs(paged=False, fp8=fp8, diffkv=diffkv)
    args["block_table"] = torch.zeros(2, 4, dtype=torch.int32)
    result = torch.zeros(4, 4, args["v"].shape[-1], dtype=torch.bfloat16)
    varlen = create_autospec(
        api.flash_attn_3.flash_attn_varlen_func, return_value=result
    )
    cache = create_autospec(api.flash_attn_3.flash_attn_with_kvcache)
    monkeypatch.setattr(api.flash_attn_3, "flash_attn_varlen_func", varlen)
    monkeypatch.setattr(api.flash_attn_3, "flash_attn_with_kvcache", cache)
    assert api.flash_attn_varlen_func(**args, fa_version=3) is result
    varlen.assert_called_once()
    assert "block_table" not in varlen.call_args.kwargs
    assert "page_table" not in varlen.call_args.kwargs
    cache.assert_not_called()


def test_unified_apis_default_to_fa2(monkeypatch):
    varlen = create_autospec(api.flash_attn.flash_attn_varlen_func)
    cache = create_autospec(api.flash_attn.flash_attn_with_kvcache)
    varlen3 = create_autospec(api.flash_attn_3.flash_attn_varlen_func)
    cache3 = create_autospec(api.flash_attn_3.flash_attn_with_kvcache)
    monkeypatch.setattr(api.flash_attn, "flash_attn_varlen_func", varlen)
    monkeypatch.setattr(api.flash_attn, "flash_attn_with_kvcache", cache)
    monkeypatch.setattr(api.flash_attn_3, "flash_attn_varlen_func", varlen3)
    monkeypatch.setattr(api.flash_attn_3, "flash_attn_with_kvcache", cache3)
    args = varlen_inputs()
    varlen.return_value = torch.empty_like(args["q"])
    api.flash_attn_varlen_func(**args)
    q = args["q"].reshape(2, 2, 4, 128)
    cache.return_value = torch.empty_like(q)
    api.flash_attn_with_kvcache(q=q, k_cache=args["k"], v_cache=args["v"])
    varlen.assert_called_once()
    cache.assert_called_once()
    varlen3.assert_not_called()
    cache3.assert_not_called()


@pytest.mark.parametrize("diffkv", [False, True])
@pytest.mark.parametrize("version,fp8", [(2, False), (3, False), (3, True)])
@pytest.mark.parametrize(
    "lengths,num_decodes,bucketed",
    [
        ([1, 5], 1, False),
        ([2, 2, 5], 2, False),
        ([1, 3, 5], 2, True),
        ([1, 3], 2, True),
        ([5, 3], 0, False),
    ],
)
def test_split_api_calls_and_request_scales(
    monkeypatch, diffkv, version, fp8, lengths, num_decodes, bucketed
):
    from vllm_metax.v1.attention.backends import flash_attn as regular
    from vllm_metax.v1.attention.backends.flash_attn_diffkv import (
        FlashAttentionDiffKVImpl,
    )

    monkeypatch.setattr(
        utils,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(
            attention_config=SimpleNamespace(flash_attn_version=version)
        ),
    )
    monkeypatch.setenv("VLLM_METAX_ENABLE_FA_SPLIT_FORWARD", "1")
    cls = FlashAttentionDiffKVImpl if diffkv else regular.FlashAttentionImpl
    impl = object.__new__(cls)
    impl.dcp_world_size = 1
    impl.__init__(
        num_heads=4,
        head_size=192 if diffkv else 128,
        scale=0.1,
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="fp8" if fp8 else "auto",
    )
    total = sum(lengths)
    decode_tokens = sum(lengths[:num_decodes])
    starts = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32
    )
    seq_lens = torch.tensor([n + 8 for n in lengths], dtype=torch.int32)
    table = torch.arange(len(lengths) * 4, dtype=torch.int32).view(len(lengths), 4)
    metadata = SimpleNamespace(
        num_actual_tokens=total,
        query_start_loc=starts,
        seq_lens=seq_lens,
        max_query_len=max(lengths),
        max_seq_len=int(seq_lens.max()),
        block_table=table,
        causal=True,
        scheduler_metadata=None,
        split_decode_num_reqs=num_decodes,
        split_decode_num_tokens=decode_tokens,
        split_decode_seq_lens=seq_lens[:num_decodes],
        split_decode_block_table=table[:num_decodes],
        split_decode_bucket_req_bounds=tuple((i, i + 1) for i in range(num_decodes))
        if bucketed
        else None,
        split_decode_bucket_query_lens=tuple(lengths[:num_decodes])
        if bucketed
        else None,
        split_decode_bucket_token_bounds=tuple(
            (int(starts[i]), int(starts[i + 1])) for i in range(num_decodes)
        )
        if bucketed
        else None,
        split_prefill_num_reqs=len(lengths) - num_decodes,
        split_prefill_query_start_loc=starts[num_decodes:] - decode_tokens,
        split_prefill_block_table=table[num_decodes:],
        split_prefill_cu_seqlens_k=torch.cat(
            (
                torch.zeros(1, dtype=torch.int32),
                seq_lens[num_decodes:].cumsum(0, dtype=torch.int32),
            )
        ),
        split_prefill_max_seq_len=int(seq_lens[num_decodes:].max())
        if num_decodes < len(lengths)
        else 0,
        mm_prefix_query_range_tensor=None,
        rswa_prefix_lens=None,
    )
    query = torch.empty(
        total + 2,
        4,
        impl.head_size,
        dtype=torch.float8_e4m3fn if fp8 else torch.bfloat16,
    )
    cache = torch.empty(
        len(lengths) * 4,
        2,
        4,
        impl.head_size + 128,
        dtype=torch.uint8 if fp8 else torch.bfloat16,
    )
    out = torch.full((total + 2, 4, 128), -1, dtype=torch.bfloat16)
    scales = torch.arange(1, len(lengths) + 1, dtype=torch.float32).view(-1, 1)
    layer = SimpleNamespace(_q_scale=scales, _k_scale=scales * 2, _v_scale=scales * 3)
    call_ranges = []
    if num_decodes < len(lengths):
        call_ranges.append((num_decodes, len(lengths), decode_tokens, total))
    if bucketed:
        call_ranges.extend(
            (i, i + 1, int(starts[i]), int(starts[i + 1])) for i in range(num_decodes)
        )
    elif num_decodes:
        call_ranges.append((0, num_decodes, 0, decode_tokens))

    observed = []

    def kernel(**kw):
        req_start, req_end, token_start, token_end = call_ranges[len(observed)]
        observed.append(kw)
        is_varlen = "k_cache" not in kw
        if is_varlen:
            assert version == 2
            torch.testing.assert_close(
                kw["cu_seqlens_k"], metadata.split_prefill_cu_seqlens_k
            )
            torch.testing.assert_close(kw["block_table"], table[req_start:req_end])
        else:
            torch.testing.assert_close(kw["cache_seqlens"], seq_lens[req_start:req_end])
            torch.testing.assert_close(
                kw["page_table" if version == 3 else "block_table"],
                table[req_start:req_end],
            )
        if version == 3:
            for name, factor in [("q_descale", 1), ("k_descale", 2), ("v_descale", 3)]:
                torch.testing.assert_close(
                    kw[name], (scales[req_start:req_end] * factor).expand(-1, 2)
                )
        if kw.get("cu_seqlens_q") is not None:
            torch.testing.assert_close(
                kw["cu_seqlens_q"], starts[req_start : req_end + 1] - token_start
            )
            assert kw["q"].shape[0] == token_end - token_start
        else:
            assert kw["q"].shape[:2] == (req_end - req_start, lengths[req_start])
        return torch.full(
            (*kw["q"].shape[:-1], 128), len(observed), dtype=torch.bfloat16
        )

    package = api.flash_attn if version == 2 else api.flash_attn_3
    vendor = create_autospec(package.flash_attn_with_kvcache, side_effect=kernel)
    varlen = create_autospec(package.flash_attn_varlen_func, side_effect=kernel)
    monkeypatch.setattr(package, "flash_attn_with_kvcache", vendor)
    monkeypatch.setattr(package, "flash_attn_varlen_func", varlen)
    assert impl.forward(layer, query, None, None, cache, metadata, out) is out
    assert len(observed) == len(call_ranges)
    if version == 3:
        varlen.assert_not_called()
    else:
        assert varlen.call_count == int(num_decodes < len(lengths))
    for idx, (_, _, token_start, token_end) in enumerate(call_ranges, start=1):
        assert torch.all(out[token_start:token_end] == idx)
    assert torch.all(out[total:] == -1)


@pytest.mark.parametrize("with_context", [False, True])
def test_fa3_dcp_context_uses_cache_api(monkeypatch, with_context):
    from vllm_metax.v1.attention.backends import flash_attn as regular

    impl = object.__new__(regular.FlashAttentionImpl)
    impl.vllm_flash_attn_version = 3
    impl.dcp_world_size = 2
    impl.num_heads = 2
    impl.head_size = 128
    impl.scale = 0.1
    impl.logits_soft_cap = 0.0
    impl.sliding_window = (-1, -1)
    impl.alibi_slopes = None
    impl.sinks = None
    impl._dcp_max_num_tokens = 0
    impl._dcp_dtype = torch.bfloat16
    q = torch.empty(4, 2, 128, dtype=torch.bfloat16)
    out = torch.empty_like(q)
    group = SimpleNamespace(
        all_gather=lambda query, dim: torch.cat((query, query), dim=dim)
    )
    monkeypatch.setattr(regular, "get_dcp_group", lambda: group)
    monkeypatch.setattr(
        regular,
        "current_workspace_manager",
        lambda: SimpleNamespace(
            get_simultaneous=lambda spec: (torch.empty(spec[0], dtype=spec[1]),)
        ),
    )
    # Isolate communication and merging; the assertion covers API routing only.
    impl.dcp_combine = lambda result, lse, group, return_lse: (
        result[:, :2],
        lse[:, :2],
    )
    monkeypatch.setattr(
        regular, "merge_attn_states", lambda output, *args: output.zero_()
    )
    metadata = SimpleNamespace(
        query_start_loc=torch.tensor([0, 1, 4], dtype=torch.int32),
        max_query_len=3,
        block_table=torch.tensor([[2, 0], [1, 3]], dtype=torch.int32),
        max_dcp_context_kv_len=7 if with_context else 0,
        causal=True,
        num_decode_reqs=1,
        num_prefill_reqs=1,
        num_decode_tokens=1,
        num_prefill_tokens=3,
        dcp_context_cu_seqlens_k=torch.tensor([0, 3, 10], dtype=torch.int32),
        dcp_context_kv_lens=torch.tensor([3, 7], dtype=torch.int32),
    )
    cache = torch.empty(4, 4, 2, 128, dtype=torch.bfloat16)
    cache_kernel = create_autospec(
        api.flash_attn_3.flash_attn_with_kvcache,
        return_value=(torch.zeros(4, 4, 128, dtype=torch.bfloat16), torch.zeros(4, 4)),
    )
    varlen_kernel = create_autospec(
        api.flash_attn_3.flash_attn_varlen_func,
        return_value=(torch.zeros_like(q), torch.zeros(2, 4)),
    )
    monkeypatch.setattr(api.flash_attn_3, "flash_attn_with_kvcache", cache_kernel)
    monkeypatch.setattr(api.flash_attn_3, "flash_attn_varlen_func", varlen_kernel)
    impl._forward_with_dcp(q, q, q, cache, cache, out, metadata)
    assert torch.all(out == 0)
    varlen_kernel.assert_called_once()
    assert "page_table" not in varlen_kernel.call_args.kwargs
    assert varlen_kernel.call_args.kwargs["k"].ndim == 3
    if with_context:
        cache_kernel.assert_called_once()
        kw = cache_kernel.call_args.kwargs
        assert kw["cache_seqlens"] is metadata.dcp_context_kv_lens
        assert kw["page_table"] is metadata.block_table
        assert kw["return_softmax_lse"] is True
        assert kw["causal"] is False
        assert kw["q"].shape[1] == 4
    else:
        cache_kernel.assert_not_called()


@pytest.mark.parametrize(
    "args,fa3_available,expected,reasons",
    [
        (dict(head_size=96), True, 2, ["96/96", "non-FP8"]),
        (dict(head_size=32, head_size_v=32), True, 2, ["32/32"]),
        (dict(head_size=512), True, 2, ["512/512"]),
        (dict(requires_alibi=True), True, 2, ["ALiBi"]),
        (dict(has_sinks=True), True, 2, ["attention sinks"]),
        (
            dict(head_size=96, requires_alibi=True, has_sinks=True),
            True,
            2,
            ["96/96", "ALiBi", "attention sinks"],
        ),
        ({}, False, 2, ["FA3 is unavailable", "test import failure"]),
        (dict(head_size=128), True, 3, []),
        (dict(head_size=80), True, None, []),
        (dict(kv_cache_block_size=24), True, None, []),
        (dict(kv_cache_dtype="fp8", has_sinks=True), True, None, []),
    ],
)
@pytest.mark.parametrize("requested", [None, 3])
def test_fa3_fallback_logs_specific_reasons_once(
    monkeypatch, caplog, args, fa3_available, expected, reasons, requested
):
    import logging
    from vllm.logger import _print_warning_once

    monkeypatch.setattr(
        utils,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(
            attention_config=SimpleNamespace(flash_attn_version=requested)
        ),
    )
    monkeypatch.setattr(api, "FA3_AVAILABLE", fa3_available)
    monkeypatch.setattr(api, "FA3_UNAVAILABLE_REASON", "test import failure")
    monkeypatch.setattr(utils.logger, "propagate", True)
    _print_warning_once.cache_clear()
    if requested is None and expected == 3:
        expected = 2
    with caplog.at_level(logging.WARNING, logger=utils.logger.name):
        for _ in range(2):
            assert utils.get_flash_attn_version(**args) == expected
    warnings = [
        r.getMessage()
        for r in caplog.records
        if "falling back to FA2" in r.getMessage()
    ]
    if expected == 2 and requested == 3:
        assert len(warnings) == 1
        assert all(reason in warnings[0] for reason in reasons)
    else:
        assert warnings == []
    _print_warning_once.cache_clear()


@pytest.mark.parametrize("requested", [None, 3])
def test_distinct_fa3_fallback_reasons_are_both_logged(monkeypatch, caplog, requested):
    import logging
    from vllm.logger import _print_warning_once

    monkeypatch.setattr(
        utils,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(
            attention_config=SimpleNamespace(flash_attn_version=requested)
        ),
    )
    monkeypatch.setattr(utils.logger, "propagate", True)
    _print_warning_once.cache_clear()
    with caplog.at_level(logging.WARNING, logger=utils.logger.name):
        assert utils.get_flash_attn_version(head_size=96) == 2
        assert utils.get_flash_attn_version(requires_alibi=True) == 2
    warnings = [
        r.getMessage()
        for r in caplog.records
        if "falling back to FA2" in r.getMessage()
    ]
    assert len(warnings) == (2 if requested == 3 else 0)
    if requested == 3:
        assert "96/96" in warnings[0]
        assert "ALiBi" in warnings[1]
    _print_warning_once.cache_clear()


@pytest.mark.parametrize("kv", ["fp8", "fp8_e4m3"])
@pytest.mark.parametrize("diffkv", [False, True])
def test_explicit_fa2_fp8_conflict_reports_reason(monkeypatch, kv, diffkv):
    from vllm_metax.v1.attention.backends import flash_attn as regular
    from vllm_metax.v1.attention.backends.flash_attn_diffkv import (
        FlashAttentionDiffKVImpl,
    )

    config = SimpleNamespace(attention_config=SimpleNamespace(flash_attn_version=2))
    monkeypatch.setattr(utils, "get_current_vllm_config_or_none", lambda: config)
    monkeypatch.setattr(regular, "get_current_vllm_config_or_none", lambda: config)
    expected = (
        "FP8 KV cache requires FA3, but flash_attn_version=2 was explicitly requested."
    )
    head_size = 192 if diffkv else 128
    backend = FlashAttentionDiffKVBackend if diffkv else Backend
    cls = FlashAttentionDiffKVImpl if diffkv else regular.FlashAttentionImpl

    # Capability probes must remain non-throwing; selection reports the reason.
    assert utils.get_flash_attn_version(head_size=head_size, kv_cache_dtype=kv) is None
    assert utils.flash_attn_supports_kv_cache_dtype(kv, head_size=head_size) is False
    assert (
        backend.supports_combination(
            head_size,
            torch.bfloat16,
            kv,
            16,
            False,
            False,
            False,
            False,
            DeviceCapability(8, 9),
        )
        == expected
    )

    impl = object.__new__(cls)
    with pytest.raises(ValueError) as error:
        impl.__init__(
            num_heads=4,
            head_size=head_size,
            scale=0.1,
            num_kv_heads=2,
            alibi_slopes=None,
            sliding_window=None,
            kv_cache_dtype=kv,
        )
    assert str(error.value) == expected


@pytest.mark.parametrize(
    "name,value",
    [
        ("alibi_slopes", torch.ones(4)),
        ("s_aux", torch.ones(4)),
        ("scheduler_metadata", torch.ones(4)),
        ("dropout_p", 0.5),
    ],
)
def test_fa3_ignores_unsupported_varlen_options(monkeypatch, name, value):
    args = varlen_inputs(paged=False)
    result = torch.zeros_like(args["q"])
    vendor = create_autospec(
        api.flash_attn_3.flash_attn_varlen_func, return_value=result
    )
    monkeypatch.setattr(api.flash_attn_3, "flash_attn_varlen_func", vendor)
    assert api.flash_attn_varlen_func(**args, fa_version=3, **{name: value}) is result
    assert name not in vendor.call_args.kwargs


def test_fa2_varlen_ignores_num_splits(monkeypatch):
    args = varlen_inputs()
    result = torch.zeros_like(args["q"])
    vendor = create_autospec(api.flash_attn.flash_attn_varlen_func, return_value=result)
    monkeypatch.setattr(api.flash_attn, "flash_attn_varlen_func", vendor)
    assert api.flash_attn_varlen_func(**args, num_splits=8) is result
    assert "num_splits" not in vendor.call_args.kwargs


@pytest.mark.parametrize("version", [2, 3])
def test_cache_shared_options_preserve_supported_parameters(monkeypatch, version):
    args = varlen_inputs()
    q = args["q"].reshape(2, 2, 4, 128)
    result = torch.zeros_like(q)
    lse = torch.zeros(2, 4, 2)
    package = api.flash_attn if version == 2 else api.flash_attn_3
    vendor = create_autospec(
        package.flash_attn_with_kvcache,
        return_value=result if version == 2 else (result, lse),
    )
    monkeypatch.setattr(package, "flash_attn_with_kvcache", vendor)
    options = dict(
        q_v=torch.ones_like(q),
        q_descale=torch.ones(2, 2),
        k_descale=torch.ones(2, 2),
        v_descale=torch.ones(2, 2),
        scheduler_metadata=torch.ones(1),
        alibi_slopes=torch.ones(4),
        s_aux=torch.ones(4),
        cu_seqlens_q=torch.tensor([0, 2, 4], dtype=torch.int32),
        cu_seqlens_k_new=torch.tensor([0, 1, 2], dtype=torch.int32),
        max_seqlen_q=2,
        rotary_seqlens=torch.tensor([4, 4], dtype=torch.int32),
        pack_gqa=True,
        attention_chunk=4,
        sm_margin=1,
        num_splits=2,
        return_softmax_lse=True,
    )
    out = torch.empty_like(result)
    answer = api.flash_attn_with_kvcache(
        q=q,
        k_cache=args["k"],
        v_cache=args["v"],
        block_table=args["block_table"],
        out=out,
        fa_version=version,
        **options,
    )
    assert (answer[0] if version == 3 else answer) is out
    kw = vendor.call_args.kwargs
    assert kw["num_splits"] == 2  # Supported by both cache APIs.
    if version == 2:
        assert kw["alibi_slopes"] is options["alibi_slopes"]
        assert kw["s_aux"] is options["s_aux"]
        for name in options.keys() - {"alibi_slopes", "s_aux", "num_splits"}:
            assert name not in kw
    else:
        assert answer[1] is lse
        assert kw["qv"] is options["q_v"]
        for name in options.keys() - {
            "q_v",
            "alibi_slopes",
            "s_aux",
            "scheduler_metadata",
        }:
            assert kw[name] is options[name]
        for name in ("alibi_slopes", "s_aux", "scheduler_metadata"):
            assert name not in kw


def test_fa3_varlen_still_rejects_paged_input_shapes(monkeypatch):
    vendor = create_autospec(api.flash_attn_3.flash_attn_varlen_func)
    monkeypatch.setattr(api.flash_attn_3, "flash_attn_varlen_func", vendor)
    with pytest.raises(ValueError, match="packed Q/K/V"):
        api.flash_attn_varlen_func(**varlen_inputs(), fa_version=3)
    vendor.assert_not_called()


@pytest.mark.parametrize(
    "cap",
    [
        DeviceCapability(8, 0),
        DeviceCapability(8, 6),
        DeviceCapability(8, 9),
        DeviceCapability(9, 0),
        DeviceCapability(10, 0),
    ],
)
def test_temporary_fa3_default_fallback_requires_fp8_device(monkeypatch, cap):
    monkeypatch.setattr(api, "FA2_AVAILABLE", False)
    monkeypatch.setattr(
        utils, "current_platform", SimpleNamespace(get_device_capability=lambda: cap)
    )
    assert utils.get_flash_attn_version(head_size=128) == (
        3 if cap == DeviceCapability(8, 9) else None
    )


@pytest.mark.parametrize(
    "cap",
    [
        DeviceCapability(8, 0),
        DeviceCapability(8, 6),
        DeviceCapability(9, 0),
        DeviceCapability(10, 0),
    ],
)
def test_temporary_fa3_device_gate_logs_fallback(monkeypatch, caplog, cap):
    import logging
    from vllm.logger import _print_warning_once

    monkeypatch.setattr(
        utils,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(attention_config=SimpleNamespace(flash_attn_version=3)),
    )
    monkeypatch.setattr(utils.logger, "propagate", True)
    _print_warning_once.cache_clear()
    with caplog.at_level(logging.WARNING, logger=utils.logger.name):
        assert utils.get_flash_attn_version(head_size=128, device_capability=cap) == 2
    assert "falling back to FA2" in caplog.text
    assert "temporarily enabled only on FP8-capable devices" in caplog.text
    assert str(cap) in caplog.text
    _print_warning_once.cache_clear()


@pytest.mark.parametrize(
    "attn_type", [AttentionType.ENCODER, AttentionType.ENCODER_ONLY]
)
@pytest.mark.parametrize(
    "dims", [(64, 64), (128, 128), (192, 192), (256, 256), (192, 128)]
)
def test_fa3_encoder_forward_uses_noncausal_varlen(monkeypatch, attn_type, dims):
    from vllm_metax.v1.attention.backends import flash_attn as regular
    from vllm_metax.v1.attention.backends.flash_attn_diffkv import (
        FlashAttentionDiffKVImpl,
    )

    monkeypatch.setattr(
        utils,
        "get_current_vllm_config_or_none",
        lambda: SimpleNamespace(attention_config=SimpleNamespace(flash_attn_version=3)),
    )
    cls = FlashAttentionDiffKVImpl if dims[0] != dims[1] else regular.FlashAttentionImpl
    impl = object.__new__(cls)
    impl.dcp_world_size = 1
    impl.__init__(
        num_heads=4,
        head_size=dims[0],
        scale=dims[0] ** -0.5,
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        attn_type=attn_type,
    )
    assert impl.vllm_flash_attn_version == 3
    starts = torch.tensor([0, 1, 4], dtype=torch.int32)
    metadata = SimpleNamespace(
        num_actual_tokens=4, query_start_loc=starts, max_query_len=3
    )
    q = torch.empty(6, 4, dims[0], dtype=torch.bfloat16)
    k = torch.empty(6, 2, dims[0], dtype=torch.bfloat16)
    v = torch.empty(6, 2, dims[1], dtype=torch.bfloat16)
    out = torch.full((6, 4, dims[1]), -1, dtype=torch.bfloat16)
    expected = torch.zeros(4, 4, dims[1], dtype=torch.bfloat16)
    varlen = create_autospec(
        api.flash_attn_3.flash_attn_varlen_func, return_value=expected
    )
    cache = create_autospec(api.flash_attn_3.flash_attn_with_kvcache)
    monkeypatch.setattr(api.flash_attn_3, "flash_attn_varlen_func", varlen)
    monkeypatch.setattr(api.flash_attn_3, "flash_attn_with_kvcache", cache)
    result = impl.forward(SimpleNamespace(), q, k, v, None, metadata, out)
    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out[:4], expected)
    assert torch.all(out[4:] == -1)
    cache.assert_not_called()
    kw = varlen.call_args.kwargs
    assert kw["causal"] is False
    assert kw["q"].shape[0] == kw["k"].shape[0] == kw["v"].shape[0] == 4
    assert kw["cu_seqlens_q"] is kw["cu_seqlens_k"] is starts
    assert "page_table" not in kw
