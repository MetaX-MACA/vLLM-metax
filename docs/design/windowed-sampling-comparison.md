# Fixed-revision sampling comparison

This review is bounded to the following source snapshots and named paths. It
does not establish absence from arbitrary third-party extensions or future
releases. The experiment archive records numbered excerpts, full-file SHA256
hashes and bounded searches checked against each repository's Git objects.

| Repository | Reviewed revision |
| --- | --- |
| vLLM | `aa4022dc8176db8f769f93445d845f8755f9b4ca` |
| vLLM-Omni | `4cbfcc3eb55597720b84339afab0842c03ac3c93` |
| MetaX base | `0962c8735ffe4795800030680311410848512b82` |

## Existing behavior

| Source location | Observed behavior | Difference from this proposal |
| --- | --- | --- |
| vLLM `sampling_params.py:296-309,419` | Standard presence, frequency, repetition and an existing generic `extra_args` extension field | No first-class request policy combining generated-token windows, eligible intervals and exemptions in these standard fields |
| vLLM `v1/sample/ops/penalties.py:10-52` | Converts the complete output histories to padded tensors; masks async placeholders before standard penalties | This processor reads only the last N generated tokens, excludes prompts and updates eligible unique coordinates |
| vLLM `model_executor/layers/utils.py:43-79` | Repetition considers prompt and output masks; frequency/presence use output counts | The proposed policy is not a drop-in semantic replacement for standard penalties |
| vLLM `v1/worker/gpu/sample/penalties.py:20-105,150-185` | V2 already has persistent GPU history statistics and a fused penalty kernel | Fused GPU penalty execution is not new; this proposal does not implement the V2 interface |
| Omni `models/yue2/yue2.py:70-110,2250-2280,2390-2448` | Request-specific `yue2_penalty_window`, repetition and phase-specific vocabulary, with grouped graph sampling | Windowed and vocabulary-aware sampling already exists in this model-specific implementation |
| Omni `models/minicpmo_4_5/codec_state.py:72-119` | Dedicated codec histories and window penalties; scales by repetition raised to each token's window frequency | Different penalty semantics and model-private state; not a general request-level combination of the three penalties |

The Omni paths in the table are under `vllm_omni/model_executor/`. DRY and other
repetition controls are also prior art, not discoveries of this work.

## Narrow addition

The addition is a reusable, explicitly loaded V1 processor exposing **one
per-request combination** of generated-token window, caller-supplied eligible
token intervals, token exemptions, sign-aware repetition once per unique token,
and additive frequency/presence penalties. Sparse execution avoids retaining a
batch-by-vocabulary state buffer when the chosen window is small.

No equivalent combination was identified in the reviewed standard sampling
paths. The claim is the combination and reusable interface, not invention of
window penalties, modality-specific vocabulary, Triton fusion, or the upstream
extension mechanism. Modalities are represented by caller-provided tokenizer
ranges; this code does not identify a token's modality automatically. There is
one generated-token window per request, not independent windows per modality.

## Integration contracts

- `v1/sample/logits_processor/interface.py:24-109` specifies live output-list
  references and removal, addition, movement ordering. The adapter follows it.
- `v1/worker/gpu_model_runner.py:3498-3540` resolves async histories immediately
  before sampling. The processor builds state in `apply`, not `update_state`.
- `v1/worker/gpu_input_batch.py:870-910,1012-1060` keeps output histories for
  non-argmax-invariant processors and fills async placeholders. An enabled custom
  processor may therefore add engine overhead even for requests without policy.
- Actual upstream loader, `SamplingParams`, OpenAI request conversion, processor
  chain and greedy `Sampler.forward` are exercised in the isolated CPU suite.
  Native greedy sampling is built from the unmodified reviewed CPU translation
  unit, not a mock replacement. This is component integration, not model serving.
- The production kernel imports `vllm.triton_utils`. Standalone HPCC component
  measurements use identical kernel functions with direct Torch/Triton imports.

The one-off source auditor is retained in the original experiment delivery,
not installed with the processor. Compare the actual code and semantics at the
fixed revisions above; keyword absence alone is not the novelty argument.
