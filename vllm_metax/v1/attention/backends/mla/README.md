# BF16 sparse MLA metadata

Uniform decode requests with no padded request rows use the existing contiguous
sequence-length prefix. This avoids uploading request-row indices and gathering
sequence lengths, including uniform multi-token decode. The optimization requires
all CPU query lengths to be equal and positive, and their total to match the
decode token count.

Variable query lengths, inactive request rows and graph token padding keep the
existing request-to-token mapping. Prefill and context-parallel metadata paths
retain their existing behavior. This reduces metadata work; no throughput gain
is claimed without a workload benchmark.
