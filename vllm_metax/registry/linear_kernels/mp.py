# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by MetaX Integrated Circuits (Shanghai) Co., Ltd. All Rights Reserved.

# -----------------------------------------------
# Note: Register MetaX mixed-precision linear kernels for out-of-tree dispatch.
#
# Affected versions: v0.21.0
# -----------------------------------------------

from vllm.model_executor.kernels.linear.mixed_precision.exllama import (
    ExllamaLinearKernel as vllm_ExllamaLinearKernel,
    MPLinearLayerConfig,
)
from vllm.model_executor.kernels.linear import register_linear_kernel
from vllm.platforms import PlatformEnum, current_platform

import torch

from vllm.model_executor.parameter import BasevLLMParameter, permute_param_layout_
from vllm.model_executor.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32,
)

import vllm_metax._custom_ops as mx_ops


def _allocate_gptq_workspaces(
    x_2d: torch.Tensor,
    qweight: torch.Tensor,
    weight_bits: int,
    group_size: int,
    desc_act: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    perm_space = torch.empty(0, device=x_2d.device)
    temp_space = torch.empty(0, device=x_2d.device)

    if weight_bits not in (4, 8) and group_size not in (64, 128):
        return perm_space, temp_space

    if desc_act:
        perm_space = torch.empty(
            x_2d.shape[0],
            x_2d.shape[1],
            dtype=torch.float16,
            device=x_2d.device,
        )

    if x_2d.dtype == torch.bfloat16:
        temp_space = torch.zeros(
            x_2d.shape[0],
            qweight.shape[1],
            dtype=torch.float32,
            device=x_2d.device,
        )

    return perm_space, temp_space


# This used for:
#  - warmup in process_weights_after_loading
#  - forward implement
def _apply_gptq_impl(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    bias: torch.Tensor | None,
    g_idx: torch.Tensor,
    use_exllama: bool,
    weight_bits: int,
    group_size: int,
    desc_act: bool,
) -> torch.Tensor:
    reshaped_x = x.reshape(-1, x.shape[-1])
    out_shape = x.shape[:-1] + (qweight.shape[-1],)

    perm_space, temp_space = _allocate_gptq_workspaces(
        reshaped_x,
        qweight,
        weight_bits,
        group_size,
        desc_act,
    )

    output = mx_ops.gptq_gemm(
        reshaped_x,
        qweight,
        qzeros,
        scales,
        g_idx,
        use_exllama,
        weight_bits,
        group_size,
        perm_space,
        temp_space,
        reshaped_x.dtype == torch.bfloat16,
    )

    if bias is not None:
        output.add_(bias)
    return output.reshape(out_shape)


def _gptq_shuffle(qweight: torch.Tensor, weight_bits: int) -> None:
    # NOTE(metax): The source operator retains the pre-v0.33 permutation input.
    # Empty device storage supplies the native null pointer without Meta dispatch.
    # Translate only known schemas; native execution errors must propagate.
    shuffle = torch.ops._C.gptq_shuffle.default
    names = [argument.name for argument in shuffle._schema.arguments]
    if names == ["q_weight", "q_perm", "bit"]:
        shuffle(
            qweight,
            torch.empty(0, dtype=torch.int32, device=qweight.device),
            weight_bits,
        )
    elif names == ["q_weight", "bit"]:
        shuffle(qweight, weight_bits)
    else:
        raise RuntimeError(f"Unsupported GPTQ shuffle schema: {shuffle._schema}")


class MacaExllamaLinearKernel(vllm_ExllamaLinearKernel):
    @classmethod
    def can_implement(cls, c: MPLinearLayerConfig) -> tuple[bool, str | None]:
        if not current_platform.is_cuda_alike():
            return (
                False,
                "Exllama is only supported on CUDA and ROCm and Maca",
            )

        if c.partition_weight_shape[1] % (32 // c.weight_type.size_bits) != 0:
            return (
                False,
                "Output features must be a multiple of the pack "
                "factor (32 / num_bits) so that we can correctly "
                "pack the zero points",
            )

        # ------------------------------------------------
        # On maca we support both float16 and bfloat16
        # ------------------------------------------------
        if c.act_type not in (torch.float16, torch.bfloat16):
            return False, "Exllama only supports float16 and bfloat16 activations"

        if c.weight_type not in cls.SUPPORTED_QUANT_TYPES:
            return (
                False,
                f"Quant type ({c.weight_type}) not supported by "
                "Exllama, supported types are: "
                f"{cls.SUPPORTED_QUANT_TYPES}",
            )

        if c.group_size <= 0:
            return (
                False,
                f"Group size ({c.group_size}) must be positive, "
                "Exllama does not support channelwise quantization",
            )

        if c.full_weight_shape[0] % c.group_size != 0:
            return (
                False,
                f"Group size ({c.group_size}) does not evenly divide"
                " the number of input features "
                f"({c.full_weight_shape[0]})",
            )

        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        c = self.config

        # For Exllama, we need to set a zero-point tensor if there is not one
        if not c.zero_points:
            self.w_zp_name = "qzeros"
            device = getattr(layer, self.w_q_name).device
            groups = c.partition_weight_shape[0] // c.group_size
            out_features = c.partition_weight_shape[1]

            if c.weight_type.has_bias():
                # if the type has a bias we have to create a zeros tensor that
                # contains the bias values repeated for each group (-1 due to
                # a bug in the original GPTQ checkpoint format leading to
                # exllama kernel adding 1 to the zero points during inference)
                # Documentation of the bug can be found here:
                #  https://garden.danieldk.eu/GPTQ-Checkpoint-Format
                zeros = torch.full(
                    (groups, out_features),
                    c.weight_type.bias - 1,
                    dtype=torch.int32,
                    device=device,
                )
            else:
                raise NotImplementedError(
                    "A 0 zero-point is not supported by Exllama due to "
                    "a bug in the original GPTQ checkpoint format leading to "
                    "exllama kernel adding 1 to the zero points during "
                    "inference"
                )
            zeros = pack_quantized_values_into_int32(zeros, c.weight_type, packed_dim=1)
            replace_parameter(layer, self.w_zp_name, zeros)

        def transform_w_q(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1, packed_dim=0)
            x_cont = x.data.contiguous()
            _gptq_shuffle(x_cont, c.weight_type.size_bits)
            return x_cont

        def transform_w_s(x):
            assert isinstance(x, BasevLLMParameter)
            permute_param_layout_(x, input_dim=0, output_dim=1)
            x.data = x.data.contiguous()
            return x.to(dtype=c.act_type)

        # Repack weights and scales for Machete
        self._transform_param(layer, self.w_q_name, transform_w_q)
        self._transform_param(layer, self.w_s_name, transform_w_s)
        if c.group_size not in (64, 128):
            return

        w_q, w_s, w_zp = self._get_weight_params(layer)
        w_g_idx = torch.empty(0, dtype=torch.int32, device=w_q.device)
        assert w_zp is not None, "Zero points are required by Exllama"

        if w_s.dtype == torch.bfloat16:
            return

        # partition_weight_shape[0] is the unpacked input width, equivalent to
        # the old qweight.shape[0] * (32 // weight_bits) warmup shape.
        warmup_x = torch.randn(
            1,
            c.partition_weight_shape[0],
            dtype=w_s.dtype,
            device=w_q.device,
        )
        _ = _apply_gptq_impl(
            warmup_x,
            w_q,
            w_s,
            w_zp,
            None,
            w_g_idx,
            True,
            c.weight_type.size_bits,
            c.group_size,
            False,
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        c = self.config

        w_q, w_s, w_zp = self._get_weight_params(layer)
        w_g_idx = torch.empty(0, dtype=torch.int32, device=w_q.device)

        assert w_zp is not None, "Zero points are required by Exllama"
        return _apply_gptq_impl(
            x,
            w_q,
            w_s,
            w_zp,
            bias,
            w_g_idx,
            True,
            c.weight_type.size_bits,
            c.group_size,
            False,
        )


register_linear_kernel(
    kernel_class=MacaExllamaLinearKernel, platform=PlatformEnum.OOT, kernel_type="mp"
)
