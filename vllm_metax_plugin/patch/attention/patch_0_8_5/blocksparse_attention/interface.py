# 2025 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import vllm
from vllm_metax_plugin.patch.hook_registry import register_patch
from vllm.logger import init_logger

logger = init_logger(__name__)
import math
from vllm.attention.ops.blocksparse_attention.interface import IS_COMPUTE_8_OR_ABOVE

IS_COMPUTE_8_OR_ABOVE = True

if IS_COMPUTE_8_OR_ABOVE:
    from .blocksparse_attention_kernel import blocksparse_flash_attn_varlen_fwd
    
def varlen_attn(self,
                q,
                k,
                v,
                cu_seqlens_k,
                cu_seqlens_q=None,
                sm_scale=None):
    """
    q, k, v: shape = (num_tokens, num_heads_q/kv, head_size).
    Support grouped attention, with `q[:, i*r:(i*r + r)]`
    is correspondent to `k[:, i]`, where `r` is the q/k ratio.
    cu_seqlens_k: shape=(batch_size + 1,),
    indicating segment of samples,
    e.g., `k[cu_seqlen[i]:cu_seqlne[i+1]]` is q of sample i
    cu_seqlens_q: shape=(batch_size + 1, ).
    Default None: same as cu_seqlens_k for prefilling or
    [0, 1, .., batch_size] for decoding.
    The only case you need to specify is when q is a mix of
    prefilling and decoding.
    sm_scale: softmax scale, default to 1/sqrt(head_size).

    return: tensor of shape as q.
    """
    # assert (
    #     IS_COMPUTE_8_OR_ABOVE
    # ), "Requires compute capability of 8 or above (Ampere or newer) to use \
    #     Triton kernel."

    sm_scale = sm_scale or 1.0 / math.sqrt(q.size(-1))

    return blocksparse_flash_attn_varlen_fwd(
        q,
        k,
        v,
        cu_seqlens_k,
        cu_seqlens_q,
        sm_scale,
        self.sparse_layout,
        block_size=self.block_size,
        q_block_size=self.q_block_size,
        max_seqlen=self.max_seqlen,
    )

vllm.attention.ops.blocksparse_attention.interface.LocalStridedBlockSparseAttn.varlen_attn = varlen_attn
register_patch("vllm.attention.ops.blocksparse_attention.interface", "LocalStridedBlockSparseAttn.varlen_attn", varlen_attn)
