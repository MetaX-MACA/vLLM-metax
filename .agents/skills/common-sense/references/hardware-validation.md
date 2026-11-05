# Chip-dependent validation scope

Collect the exact chip type, SDK, device selection and reported compute capability
before choosing validation configurations. CUDA capability numbers and upstream feature
flags do not establish support on MACA. Verify the selected installed component too.

- **C5xx:** do not execute FP8, MXFP8, MXFP4, NVFP4 or other unsupported low-bit compute
  or cache paths. Use supported INT8, BF16 and FP16 cases, subject to the actual kernel's
  shape/layout and parallel restrictions.
- **Unseen/unknown chip:** default to INT8, BF16 and FP16. Enable FP8-family or FP4-family
  validation only after the user explicitly states support for that chip and the
  installed components/selected implementation permit the concrete path.
- Distinguish checkpoint storage from runtime weights, activations, attention cache and
  indexer cache. An FP8 checkpoint converted to supported BF16/INT8 execution is not
  evidence that the chip supports FP8 arithmetic or an FP8 KV cache.

Complete source adaptation of unavailable branches against the confirmed target. Do
not delete them, invent capability claims, or run an unsupported kernel to mark a file
done. Bind acceptance tasks and support-matrix rows to the available-chip scope. Record
each unavailable branch with files, exact feature/quantization, chip, reason and resume
condition using `coverage-put`; keep its status `unverified`. Omission from the test
matrix without this record hides incomplete hardware coverage.

A file's final completed label is scoped to its declared supported configurations.
The run may finish when all files are source-adapted and required available-chip
acceptance passes, while these other branches remain explicitly unverified. Do not
represent that result as universal FP8/FP4 validation or support.

## Production follow-up

TODO: implement a maintained chip-type/compute-capability table and production capability
checks as supported chip families are supplied. This skill/tooling refactor does not
add that table, infer a future chip's capabilities, or modify production quantization
selection. Persist the missing capability facts and the configurations to revisit.
