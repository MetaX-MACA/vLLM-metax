---
name: patch-upgrade
description: Review every MetaX monkey patch with removal as the objective, using target upstream fixes, supported extension points and adapted local modules; minimize and justify any unavoidable retained patch.
---

# Patch Upgrade

Apply [common-sense](../common-sense/SKILL.md). This skill concerns `vllm_metax/patch/`;
normal model/backend/kernel adaptations do not inherit patch headers or audit requirements.
Read the patch README, templates and existing `AUDIT.md` for repository conventions.

## Review every patch with removal as the objective

In a full upgrade, perform the complete patch pass **after all framework and model
source adaptation**. The coordinator may permit an early narrowly scoped blocking
repair with an explicit reason; re-review that patch during the final patch phase.

Inventory each replacement and activation path, including disabled files, direct
assignments, registry mutation, `sys.modules` redirects, aliases and aggregate imports.
Classify infrastructure separately. One file can contain multiple patch decisions.
For every patch establish the actual target, original purpose, current target CUDA
behavior, MACA restriction and all bound callers before deciding:

1. **Remove** if upstream fixed the issue, the path disappeared, the requirement no
   longer applies, or the completed local adaptation makes it redundant.
2. **Replace with an existing supported extension point** or the appropriate local
   implementation, then remove the monkey patch and overlapping activation.
3. **Retain only the irreducible difference** when the first two choices cannot preserve
   required behavior. Explain why they failed. Simplify a full replacement to a narrow
   wrapper/hook where possible, preserving unaffected upstream behavior.

Challenge every retention. An old adaptation, disabled state or historical PR link is
not a reason to keep or delete it; verify the actual target revision and trigger.
Do not merely rename or relocate a monkey patch and call it removed. If removal needs
a normal module repair, track that repair and consumer revalidation before final model
validation. Never trade required MACA correctness for a smaller patch count.

## Preserve exact bindings and semantics

- Resolve dynamic exports, inheritance, decorators and aliases to the real target.
  A failed static lookup does not prove removal. Check old `from ... import ...`
  bindings, descriptor behavior, repeated activation and fresh-process import order.
- Use the supported registry/CustomOp/hook for only the platform entries that need it;
  preserve unrelated platforms and explicit inputs. A small wrapper delegates unchanged
  behavior to the target implementation.
- Follow template decorator semantics. `allow_missing=True` is only for an intentional
  added compatibility attribute, never a renamed target or typo.
- Preserve upstream validation, model branches, cleanup, lazy initialization and Triton
  decorator order. Inspect parameter meanings, not just matching names.
- Read [review-patterns.md](references/review-patterns.md) for quantization arguments,
  tokenizer behavior, regional ordering, kernel and communication restrictions.

## Explain retained patches briefly

Keep license/copyright and the repository's dedicated header on each Python file,
including infrastructure. Use concise factual text:

```python
# -----------------------------------------------------------------------------
# Note: Concrete MACA restriction and why the target upstream path is insufficient.
#
# Affected versions: Evidence-backed scope and reviewed target revision.
#
# Remove at: Verifiable upstream/component condition that removes this restriction.
# -----------------------------------------------------------------------------
```

Do not add long algorithm essays or mandatory examples to every patch. Preserve useful
existing/upstream comments; place detailed invariants, comparison examples and evidence
in `AUDIT.md` or tests. Infrastructure notes describe infrastructure without inventing
an upstream defect. Each retained patch needs a current reason and removal condition.

## Patch acceptance

Verify removal retires activation and stale aliases; callers reach the intended original
or supported local path. Verify retained patches bind the exact intended target in fresh
processes. Use discriminating component cases for configuration overrides, state/batch
ordering, quantization, layouts, kernel restrictions or descriptor behavior as applicable.
Complete-model checks still wait for the coordinator's final validation phase.

Update the existing patch audit for every entry: remove/replace/retain outcome, target
evidence, why removal was or was not possible, remaining minimal difference, activation
and caller checks, removal condition and shared evidence links. Keep unrelated findings
outside the patch audit. Source completion requires this final removal-first review,
including patches repaired earlier.
