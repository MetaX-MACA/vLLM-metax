# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# -----------------------------------------------------------------------------
# Note: transformers >= 5.19.0 changed the video-processing contract of
#       ``BaseVideoProcessor``. ``preprocess()`` now pops ``do_convert_rgb`` out
#       of the kwargs (video_processing_utils.py) and hands it to
#       ``_prepare_input_videos()``, which performs the RGB conversion, and only
#       afterwards calls ``self._preprocess(videos=videos, **kwargs)``.
#
#       vLLM's vendored ``MiniMaxM3VLVideoProcessor._preprocess``
#       (vllm/transformers_utils/processors/minimax_m3.py) was adapted from the
#       older contract and still declares ``do_convert_rgb`` as a required
#       positional parameter, so the call always raises
#
#           TypeError: MiniMaxM3VLVideoProcessor._preprocess() missing 1
#           required positional argument: 'do_convert_rgb'
#
#       The vendored function body never reads ``do_convert_rgb`` (RGB
#       conversion is the caller's job in both contracts), so supplying a
#       default restores the previous behaviour without changing the output.
#
#       The failure is not quantisation- or platform-specific, but it is fatal:
#       it happens during ``determine_available_memory()`` -> ``profile_run()``
#       -> ``get_dummy_encoder_profile_inputs()`` because video is the modality
#       with the largest token budget for MiniMax-M3, so every worker raises and
#       the engine never becomes ready - even for text-only test suites.
#       Reproduced on C600-U with MiniMax-M3-FP8
#       (run_20261010_141519/20261010_1415).
#
# Purpose: keep MiniMax-M3 (and MiniMax-M3-FP8) serving startable on MetaX when
#     the installed transformers version pops ``do_convert_rgb`` before
#     ``_preprocess`` is called.
#
# Affected versions: vLLM 0.29.0 / 0.29.1.dev0 (98dff2a81d) with transformers
#     5.19.0, verified 2026-10-10 on C600-U (MXC600-U).
#
# Remove at: Upstream vLLM no longer requires ``do_convert_rgb`` in the vendored
#     MiniMaxM3VLVideoProcessor._preprocess signature (i.e. it follows
#     transformers' BaseVideoProcessor contract again).
# -----------------------------------------------------------------------------

from vllm.transformers_utils.processors.minimax_m3 import MiniMaxM3VLVideoProcessor

from vllm_metax.patch.utils import patch

_TARGET_MODULE = "vllm.transformers_utils.processors.minimax_m3"

# Captured before the decorator below installs the replacement, so the wrapper
# can delegate to the vendored implementation instead of copying its body.
original_preprocess = MiniMaxM3VLVideoProcessor._preprocess


@patch(_TARGET_MODULE, "MiniMaxM3VLVideoProcessor._preprocess")
def _preprocess(self, videos, *args, **kwargs):
    # /-------------------- MetaX Modification --------------------\
    # 上游签名: def _preprocess(self, videos, do_convert_rgb, do_resize, ...)
    # transformers >= 5.19.0 在本函数被调用前就把 do_convert_rgb pop 掉(交给了
    # _prepare_input_videos), 该必填参数永远缺失, 于是 TypeError 让引擎起不来。
    # 上游函数体不使用这个参数, 这里补默认值再转调原实现; 若调用方(较新的
    # transformers)仍然传了该参数, setdefault 会原样保留。
    kwargs.setdefault("do_convert_rgb", True)
    # \-------------------- MetaX Modification --------------------/
    return original_preprocess(self, videos, *args, **kwargs)
