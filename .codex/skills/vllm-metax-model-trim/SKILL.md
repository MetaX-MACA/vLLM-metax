---
name: vllm-metax-model-trim
description: Trim large-model configurations into portable directories for vllm serve --load-format dummy smoke tests on limited MetaX GPUs. Preserve relevant Dense/MoE, MLA, sparse or hybrid attention, multimodal, and MTP structures while reducing depth. Use for functional testing, not real-weight pruning, accuracy evaluation, or model compatibility upgrades.
---

# vLLM-MetaX Model Trimming

Create a portable, reduced model directory that preserves the execution paths under test.
`--load-format dummy` skips real checkpoint reads but still allocates parameters, runs
applicable quantization postprocessing, and consumes GPU memory. Random outputs do not
retain the original model's capabilities. Success does not establish real-weight loading,
accuracy, representative routing distributions, or original-model performance.

## Establish inputs and coverage

- Determine the source and output directories, available GPUs/free memory, TP/PP/EP,
  quantization, context length, and target features from the conversation and existing
  configurations. Ask only for missing inputs that affect trimming. Without explicit
  feature targets, preserve distinct backbone layer types and list MTP/DSpark separately
  as optional coverage.
- For configuration-only requests, do not allocate GPUs. When functional validation is
  requested, proceed through service and request tests. Check occupancy if available GPUs
  are unspecified; do not terminate other workloads to make room.
- Record the actual Python, vLLM, vllm_metax, and Transformers versions and import origins.
  Inspect model registration, configuration classes, constructors, and dispatch conditions.
  Workspace source may differ from runtime source. This skill does not require an
  environment upgrade or automatically invoke the upgrade skills' adaptation audits.
- Existing examples are under `tools/batched_test/models/`, not a root-level `batched_test/`.
  Read a relevant example's `config.json`, documentation, and launch script first. Avoid
  reading large tokenizer JSON files to understand architecture. Recheck example paths,
  environment variables, and historical success records against the current environment.

## Design the reduction

Read the relevant sections of the [structure guide](references/structures.md).
For unfamiliar architectures, derive field semantics from their actual consumers rather
than applying another model family's field conventions.

1. Inventory each submodel's depth, layer types, attention/cache types, Dense/MoE boundaries,
   quantization, sharing/references between layers, and optional prediction heads. Create
   a small table mapping each target path to retained layers/dependencies and a triggering request.
2. **Reduce depth first; preserve width, expert counts, head/LoRA dimensions, and quantization
   where possible.** Select the fewest layers that cover the targets and satisfy dependencies.
   A homogeneous model can start with 1–2 layers. For heterogeneous models, do not blindly
   keep the first N layers or set every `num_hidden_layers` field to the same value.
3. Record the original-to-new layer mapping. This maps structural references; it does not
   transfer real weights. Update layer arrays, periodic/offset rules, sharing relationships,
   quantization module paths, MTP indices, and cross-layer references together. Determine
   which fields the implementation reads; descriptive JSON fields may not control execution.
4. If memory is still insufficient, reduce runtime context, concurrency, and cache budgets
   before considering fewer experts or narrower dimensions. Check sharding, quantization
   groups, and kernel constraints before changing shapes. Record the original shapes and
   paths no longer covered. `gpu_memory_utilization` sets a budget; it does not shrink
   parameters, and an overly small budget can prevent startup.

## Generate a portable directory

- Default to `tools/batched_test/models/<model-name>-dummy-<N>layers/`. Choose a new name if
  it already exists, or update it as explicitly requested. Preserve the source model and
  the user's existing examples; do not modify the source configuration in place.
- Deep-copy the original JSON and make explicit field edits, preserving unknown fields.
  Use configuration-class `to_dict()` output to inspect normalization, not to overwrite
  the source JSON and potentially lose extension metadata.
- Copy only required assets: `config.json`, tokenizer data/vocabulary/merges/SentencePiece
  files, `tokenizer_config.json`, special tokens, chat templates, and applicable generation
  or processor configurations. See the structure guide for multimodal and custom-code
  requirements. Dereference required file symlinks during copying to keep the result portable.
- Omit weight shards. Weight indices are usually unnecessary; retain an index only when
  actual configuration discovery code consumes its metadata, and document its purpose and
  the absence of real shards. An index alone does not justify downloading weights. Do not
  classify files solely by suffix: a tokenizer `.model` file is not a model checkpoint.
- Preserve the active embedded or standalone quantization configuration. Do not remove
  quantization to make a quantized model pass.
- Document the source, before/after field values, layer mapping, retained and omitted paths,
  required files, environment, runnable serve command, and test results in the output
  directory. Replace stale absolute paths from existing examples.

## Validate and deliver

Follow the [validation guide](references/validation.md) within the requested scope:

1. Check field dependencies and asset completeness. Load configurations/tokenizers/processors
   offline and inspect normalized configurations and actual model registration. CPU
   configuration loading does not establish successful GPU model construction.
2. When runtime validation is requested, start `vllm serve --load-format dummy` on the selected
   GPUs. Check health, then send requests exercising prefill and multiple decode steps.
   Add long-input, modality, parallel, or speculative requests for the target features and save logs.
3. Classify failures as trimming/configuration or missing-dependency issues, insufficient
   resources, dummy initialization/quantization postprocessing limitations, or implementation
   defects. Retry only evidence-backed adjustments. If no valid configuration or available
   resources satisfy the target, deliver the generated artifacts and concrete blocker.
   Do not claim success by removing target paths, shrinking indefinitely, or modifying kernels.
4. Report configuration checks, startup, generation, and additional branches separately.
   Mark unexecuted GPU checks as unverified. Clean up services and workers started for this
   task and record abnormal shutdowns. Provide artifact paths, usage, and coverage limits.
