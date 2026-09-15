# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.
import torch
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.modular_kernel import (
    FusedMoEPrepareAndFinalize,
)

from vllm.model_executor.layers.fused_moe.all2all_utils import (
    get_ep_all2all_manager,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    make_moe_prepare_and_finalize_naive_dp_ep,
    make_moe_prepare_and_finalize_no_dp_ep,
)
from vllm.platforms import current_platform
from vllm.utils.import_utils import (
    has_deep_ep,
)

logger = init_logger(__name__)

if has_deep_ep():
    from vllm_metax.model_executor.layers.fused_moe.prepare_finalize.deepep_ll import (
        DEEPEP_QUANT_BLOCK_SHAPE,
        MacaDeepEPLLPrepareAndFinalize,
    )


def maybe_make_prepare_finalize(
    moe: FusedMoEConfig,
    quant_config: FusedMoEQuantConfig | None,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    allow_new_interface: bool = False,
    use_monolithic: bool = False,
    eep_stage: bool = False,
) -> FusedMoEPrepareAndFinalize | None:
    # NOTE(rob): we are migrating each quant_method to hold the MK
    # in all cases. The allow_new_interface=False flag allow us to fall
    # back to the old method for methods that have not yet been migrated.
    #
    # In old method:
    #   * maybe_init_modular_kernel() calls this function. If we are
    #     using no Dp/Ep or naive all2all, we return None this function
    #     returns None and no ModularKernelMethod is created. If non-naive
    #     all2all is used, this returns a PrepareAndFinalize object and
    #     a ModularKernelMethod is created.
    # In new method:
    #   * maybe_make_prepare_finalize() is called from the oracle. We
    #     always return a PrepareAndFinalize object and the quant method
    #     holds the ModularKernel.
    if not moe.moe_parallel_config.use_all2all_kernels:
        if not allow_new_interface:
            return None

        # For DP/TP case, fall back to naive P/F.
        if moe.moe_parallel_config.dp_size > 1:
            logger.info_once(
                "Detected DP deployment with no --enable-expert-parallel. "
                "Falling back to AllGather+ReduceScatter dispatch/combine."
            )
            all2all_manager = get_ep_all2all_manager(eep_stage)
            return make_moe_prepare_and_finalize_naive_dp_ep(
                is_sequence_parallel=moe.moe_parallel_config.is_sequence_parallel,
                num_dispatchers=all2all_manager.world_size,
                use_monolithic=use_monolithic,
            )
        else:
            return make_moe_prepare_and_finalize_no_dp_ep(use_monolithic)

    all2all_manager = get_ep_all2all_manager(eep_stage)

    prepare_finalize: FusedMoEPrepareAndFinalize | None = None

    # / ---------------------- Metax Modification ----------------------- \
    # Note: metax only support:
    #   * DeepEP low-latency kernels
    #   * All2All kernels with naive DP/EP
    # \ ----------------------------------------------------------------- /
    if moe.use_deepep_ll_kernels:
        assert quant_config is not None
        global_to_physical = physical_to_global = local_expert_global_ids = None
        if routing_tables is not None:
            (
                global_to_physical,
                physical_to_global,
                local_expert_global_ids,
            ) = routing_tables
        all_to_all_args = dict(
            max_num_tokens_per_dp_rank=moe.max_num_tokens,
            token_hidden_size=moe.hidden_dim,
            num_ep_ranks=all2all_manager.world_size,
            num_global_experts=moe.num_experts,
            num_local_experts=moe.num_experts // all2all_manager.world_size,
        )
        handle = all2all_manager.get_handle(all_to_all_args)

        # Note: We may want to use FP8 dispatch just to reduce
        # data movement.
        use_fp8_dispatch = (
            quant_config.quant_dtype == current_platform.fp8_dtype()
            and quant_config.block_shape == DEEPEP_QUANT_BLOCK_SHAPE
        )

        prepare_finalize = MacaDeepEPLLPrepareAndFinalize(
            handle,
            max_tokens_per_rank=moe.max_num_tokens,
            num_dispatchers=all2all_manager.world_size,
            use_fp8_dispatch=use_fp8_dispatch,
            global_to_physical=global_to_physical,
            physical_to_global=physical_to_global,
            local_expert_global_ids=local_expert_global_ids,
        )

    elif moe.use_ag_rs_all2all_kernels and allow_new_interface:
        prepare_finalize = make_moe_prepare_and_finalize_naive_dp_ep(
            use_monolithic=use_monolithic,
            is_sequence_parallel=moe.moe_parallel_config.is_sequence_parallel,
            num_dispatchers=all2all_manager.world_size,
        )

    return prepare_finalize
