# Trimming a Real Checkpoint

Use this guide only when the requested output must load source weights without
`--load-format dummy`. A reduced-depth model with genuine retained weights is a new
checkpoint: removed layers change its computation, so successful loading does not
preserve the original model's accuracy.

## Establish the checkpoint contract

- Locate every source shard and the actual checkpoint index or loader metadata. Inspect
  tensor names, shapes, dtypes, and counts without loading the whole checkpoint into RAM.
  Resolve symlinks for the final artifact. Stop if shards are missing or unreadable.
- Inspect the runtime model's `load_weights`, weight-name mapper, packed/fused projection
  mappings, expert loader, quantization loader, tied weights, and MTP loading rules. Map
  **source checkpoint names to target checkpoint names**, not to the names of internal
  fused parameters unless the model's loader explicitly expects those names.
- Decide which non-layer tensors are required: embeddings, final norms, output head,
  projectors, encoders, resamplers, scales, biases, and auxiliary heads as applicable.
  Decide whether optional MTP/DSpark weights are included based on the target configuration.
  For every retained layer, include all of its required tensors, including quantization
  scales/zero points/metadata and MoE expert tensors. Inspect tensor consumers rather than
  assuming every file or name prefix has the same meaning across model families.
- Keep shapes and dtypes compatible with the target configuration, quantization format,
  TP/PP/EP loader, and any packed group alignment. If reducing width, vocab size, expert
  count, or other dimensions requires tensor slicing or requantization, design and verify
  that transform separately; name remapping alone is insufficient. Prefer depth-only
  trimming when preserving the original tensor values is the goal.

## Storage shards and TP rank partitions

- Establish separately the checkpoint's storage layout, any saved TP/EP partitions, and the
  requested runtime TP. A Safetensors index split across many files can still contain full
  tensors. File count and statements such as “the original model uses TP8” do not establish
  that its checkpoint is pre-partitioned; inspect tensor shapes, metadata, and loader logic.
- With full tensors, a compatible loader can partition the same reduced-depth checkpoint
  at load time for TP2 or TP8, subject to its dimensional, quantization, kernel, and resource
  constraints. Removing layers reduces parameter memory; it does not itself change the
  remaining tensors' partition axes or require a TP change. Structural compatibility is
  not proof of successful loading or generation at every TP.
- With rank-local files, use the matching TP unless the loader explicitly supports
  re-sharding. Changing TP8 to TP2 requires reconstructing/repartitioning the retained
  tensors or converting from a full-tensor checkpoint; renaming files or selecting two
  ranks is insufficient. Preserve replicated tensors, expert ownership, packed quantized
  values, and their scales according to the actual converter/loader contract.
- Keep extraction/layer renaming and TP conversion as separate reproducible operations.
  A TP-only change should reuse an already validated full-tensor trimmed checkpoint when
  available; it should not force a new layer selection or another depth reduction.
  Do not assume attention, expert, and embedding partitions all follow the same rule.
- Document which deliverable contains full tensors and which contains fixed-rank partitions,
  the intended TP of each partitioned set, and which TP values were actually validated.
  Keep different formats in separate loader discovery directories. Preserve both formats
  only when useful and storage permits; dual-format output is not a requirement.

## Build the output

Generate or adapt a reproducible extraction script from the inspected checkpoint format,
target model loader, and explicit layer mapping, then execute it. Parameterize source and
output paths, refuse accidental overwrite, and save the script and its invocation with the
deliverable. A written procedure alone does not produce a real-weight checkpoint. Do not
claim that one generic name substitution works for all model families.

1. Make an explicit, one-to-one mapping for retained backbone, encoder/decoder, vision,
   and prediction layers. Replace only the layer-index path segment belonging to the
   relevant module namespace. For example, a source `model.layers.6.*` selected as new
   layer 1 becomes `model.layers.1.*`; a numeric expert ID inside that name stays 6 if
   it identifies expert 6. Keep unchanged global tensors under their original names.
2. Inventory output keys before writing. Reject duplicate target names, tensors from
   removed layers, missing retained-layer tensors, or ambiguous aliases. Distinguish
   intentional tied/shared parameters from accidental key collisions.
3. Read and write shards incrementally. Preserve tensor values and dtypes; avoid loading
   the full model into host or GPU memory. For safetensors, write valid safetensors shards
   and regenerate `model.safetensors.index.json` from the **actual output files** if the
   result is sharded. Every indexed key must exist exactly once, and every written tensor
   must appear in the index. Keep `metadata.total_size` consistent with tensor byte sizes.
   For other formats, follow the loader's actual discovery and index conventions. Do not
   copy the original index after changing layer names or shard contents.
4. Write to a separate staging directory and promote only after completeness checks.
   Preserve the source checkpoint. Record source-to-target layer and tensor-name rules,
   source file provenance, output shard list, tensor count, byte total, and omitted paths.
   Keep an output directory with only the intended configuration and checkpoint files so
   the loader cannot discover stale or extra shards.

## Verify real-weight loading

- Reopen every output shard and index offline. Confirm unique names, shape/dtype agreement,
  valid layer references, and that copied tensors contain the source values after the
  documented rename or transform. Compare full tensor hashes or bytes against the source
  for unchanged tensors; record and verify any intentional transform. Check source-to-output
  coverage for each retained layer, including expert and quantization tensors.
- Inspect the target loader's return value or loading logs for required parameter coverage.
  Some loaders skip unexpected tensors or initialize missing tensors without a fatal error;
  startup alone cannot establish complete loading. If coverage is not observable, add
  temporary diagnostic instrumentation or state this limitation explicitly.
- Start the service with a real checkpoint load format and **without** `--load-format dummy`.
  Use the same intended quantization and TP/PP/EP settings. Check health, run a prefill
  and multi-step decode request, and trigger requested optional branches. Save the command
  and logs. If the loader reports missing, unexpected, or incompatible tensors, fix the
  checkpoint or target configuration; do not silently initialize them at random or
  remove target paths to claim success.
- Report three distinct outcomes: checkpoint integrity, parameter loading coverage, and
  runtime request success. Without an executable GPU test, deliver the checkpoint as
  **unverified for runtime loading**, even if offline checks pass.
