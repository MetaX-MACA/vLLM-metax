# SPDX-License-Identifier: Apache-2.0

import pytest

ops = pytest.importorskip(
    "vllm_metax.model_executor.layers.quantization._python_api_ops",
    reason="mctlassEx / MetaX backend not available",
)


class _TensorShape:
    def __init__(self, shape):
        self.shape = shape


@pytest.mark.parametrize(
    ("a_shape", "out_shape"),
    [
        ((4, 24), (4, 16)),
        ((4, 16), (4, 24)),
        ((4, 24), (4, 24)),
    ],
)
def test_mctlass_w8a8_scaled_mm_rejects_unaligned_shape(
    a_shape, out_shape,
):
    with pytest.raises(ValueError, match="K and N to be multiples of 16"):
        ops._check_mctlass_w8a8_scaled_mm_shape(
            _TensorShape(a_shape), _TensorShape(out_shape)
        )


def test_mctlass_w8a8_scaled_mm_accepts_aligned_shape():
    ops._check_mctlass_w8a8_scaled_mm_shape(
        _TensorShape((4, 16)), _TensorShape((4, 32))
    )
