# MACA GPTQ linear kernels

The Exllama implementation uses vLLM 0.33's three weight tensors and configuration
without activation-ordering state. Its internal adapter supplies the empty device
group-index sentinel (a null native pointer) and workspaces required by the existing local eleven-argument
GPTQ operator. The public target interface remains unchanged. A meta tensor would select Meta
dispatch instead of the real CUDA/MACA kernel, so the sentinel stays on the weight
device with zero elements.

Post-load weight shuffling accepts the known two-argument upstream schema or the
three-argument MACA permutation schema. Unknown schemas and native execution errors
are reported directly. The adapter can be removed when the local operator adopts
the target schema. Source-kernel FP16/BF16, 4/8-bit and group-size 64/128 validation
is separate from MCOPLIB or distributed support.

The Exllama types are symmetric `uint4b8` and `uint8b128`. Explicit stored
zero points for these types carry their fixed bias. Target AutoGPTQ selects
these symmetric types; asymmetric compressed-tensors types (`uint4`/`uint8`)
remain rejected by this kernel's capability check.
