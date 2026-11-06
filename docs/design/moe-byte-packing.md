# Experimental lossless MoE dispatch packing

Set `VLLM_METAX_EXPERIMENTAL_MOE_BYTE_PACK=1` on **every participating rank**
to opt in. The default is disabled. Only `MacaAgRsAll2AllManager.dispatch`
changes; router-logit dispatch, combine and the existing multi-node collective
fallback remain unchanged.

## Wire contract

The Triton kernels copy raw bytes from hidden states, weights, IDs and ordered
extra tensors into rows of one `uint8` tensor. One `all_gatherv` call gathers
the rows, then unpack allocates fresh typed output tensors. There are no numeric
casts: large integer IDs, NaN bit patterns and signed zero are preserved.
Output order and optional extras follow the existing dispatch API.

The experimental route requires four ranks, nonuniform nonnegative token
counts, at most eight tokens on any rank and at most 8260 bytes per token.
Uniform, two-rank and larger cases retain the previous path. These are a
conservative HPCC measurement envelope, not hardware-independent thresholds.
At most eight tensors (including hidden states, weights and IDs) are packed.
Supported metadata includes rank 1-4, nonempty trailing dimensions, bool,
uint8, int8/16/32/64 and float16/bfloat16/float32/float64.

As in the existing collective, all ranks must agree on sizes, non-token shapes,
dtypes, tensor order and the environment setting. Routing never depends on
rank-local strides or local row count. Local tensors must be strided, on one
GPU, have matching row counts, nonnegative strides and no autograd/conjugate/
negative view state. Unsupported local storage raises rather than choosing a
different collective schedule on one rank.

Kernels execute on the caller's current stream and rely on the existing gather
stream-ordering contract. There are no persistent buffers or explicit host
synchronizations. Pack, gather and fresh unpack outputs do add temporary memory;
this is fewer collective launches, not compression or lower wire payload.

## Tests

In an installed target runtime, reserve the required devices before execution:

```bash
python -m pytest -q --confcutdir=tests/distributed tests/distributed/test_moe_byte_pack.py
torchrun --standalone --nproc-per-node=4 -m pytest -q --confcutdir=tests/distributed tests/distributed/test_moe_byte_pack.py -k distributed
```

The distributed test composes the production pack/unpack with actual ordered
broadcast collectives for uneven gather. Cross-stream and changed-input graph
tests cover local kernels separately. Reserve devices before running torchrun.

See [component measurements](component-validation.md) for benchmark results,
regressions and temporary memory costs.
