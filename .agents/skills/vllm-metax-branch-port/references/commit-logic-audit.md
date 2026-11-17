# Per-commit logic audit

Read this reference for every source commit that is marked `PORT_REQUIRED`, selected by
the user for migration or otherwise expected to produce an adaptation patch.

The purpose is to prove that the reviewer understands what the commit changes, why the
change matters, whether the target still has the same logic problem, and how the adapted
logic should behave. A diff summary or a statement that "the target is missing the
patch" is insufficient.

## Required section

Create one section using the full source SHA:

````markdown
## Commit logic audit: <full-source-sha>

- Source subject: `<subject>`
- Original author: `<name> <email>`
- Files and symbols: `<paths and functions/classes>`
- Target counterpart: `<paths and symbols>`
- Decision: `PORT_REQUIRED`

### Original intent

Explain the concrete problem or capability and why the source author made this change.

### Logic walkthrough

Describe the actual execution path:

1. Entry point and caller conditions.
2. Inputs, state and configuration involved.
3. Decision points and invariants.
4. Operation changed by the commit.
5. Outputs, side effects and error behavior.

### Root cause and semantic change

State the defect, bottleneck or missing contract. Explain exactly which value, state,
shape, ordering, dispatch selection or API contract changes and why that changes the
result.

### Target applicability

Identify the target implementation and explain why the same logic requirement remains,
or which target contract requires a different implementation. Cite concrete symbols and
code paths rather than only file names.

### Adaptation plan

Describe the target-side change:

- files and symbols to update;
- semantics that must be preserved;
- target APIs or layouts that require adaptation;
- dependencies and ordering constraints;
- unrelated behavior that must not change.

### Explanation example

Write a simple example for the user. It should make the changed behavior understandable
without requiring the full diff. This is explanatory material, not validation evidence.
Numbers and exact observed outputs are optional; a conceptual scenario, pseudocode or
small input/output comparison is acceptable.

```text
Scenario: <simple input, configuration, request, data shape or code path>
Current target: <how the target behaves without this change>
Source intent: <what the source commit changes>
After adaptation: <how the target should behave>
Why: <how the changed logic produces that result>
```

Choose a simple form that matches the commit:

- correctness fix: a small input or state transition that previously failed;
- performance fix: a workload and the redundant work or synchronization removed;
- model adaptation: a model/config/API shape and the previous mismatch;
- attention/cache fix: a simplified shape/layout case;
- registry/quantization fix: a dispatch or weight-layout example;
- build fix: a dependency/compiler/API combination.

If numbers are used, label them as illustrative unless actual measured evidence is
recorded in the validation section. Do not present the explanation example as a passing
test or proof of numerical equivalence.

### Risks and counterexamples

Record edge cases, alternative target implementations, performance risks, unsupported
configurations and any assumption that could invalidate the adaptation.

### Validation plan

List the focused static, unit, component, e2e or benchmark evidence needed to prove the
adapted behavior. Connect each check to the example where possible.
````

If the reviewer cannot explain the logic and give a simple user-facing example, the
commit cannot be `PORT_REQUIRED`; use `NEEDS_REVIEW` and state the missing information.
