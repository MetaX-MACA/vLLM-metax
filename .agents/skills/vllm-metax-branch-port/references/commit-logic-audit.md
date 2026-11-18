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
- Child intent ID: `<stable-intent-id-or-source-commit>`
- Target counterpart: `<paths and symbols>`
- Decision: `PORT_REQUIRED`
- Selection and basis: `<current user instruction or approved scope>`
- Implementation status: `<reviewing/in-progress/committed/merged/blocked>`
- Human review: `<not-required/awaiting-user-confirmation/confirmed>`
- Review reasons and requested confirmation: `<specific risks and decision>`

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
code paths rather than only file names. A missing source patch is not evidence that the
target needs the change.

### Target-first applicability proof

Record the falsifiable target-side checks:

- current target entry point, callers, callees and owned state;
- target code and history searches used to find an equivalent mechanism;
- exact evidence that no equivalent implementation exists, or the replacement the target
  already uses;
- every source assumption that must still hold, and whether it does;
- the observable target gap if the source change is omitted;
- why the proposed target mechanism is the smallest appropriate change.

### Falsification evidence

Define the behavior-level check before implementing the patch:

- input/configuration and final observable behavior;
- expected failure on the unmodified target, or an explicit non-runtime contract when
  execution is impossible;
- the test or measurement that must pass after the patch;
- why helper flags, intermediate shapes or copied patch text are insufficient proof.

For a correctness fix, the final evidence should include a fail-before/pass-after check.
For a performance change, record a paired target baseline and the workload that makes the
claim meaningful. For an API adaptation, exercise the real target caller.

### Adaptation plan

Describe the target-side change:

- files and symbols to update;
- semantics that must be preserved;
- target APIs or layouts that require adaptation;
- dependencies and ordering constraints;
- unrelated behavior that must not change.

### Final rebase re-audit

After rebasing onto the latest target baseline, record:

- the new full target SHA;
- whether target code, APIs or tests changed the applicability conclusion;
- which source assumptions remain valid;
- which patch portions became redundant;
- whether the falsification check still fails without the patch and passes with it.

A textual conflict-free rebase does not prove that the patch is still needed.

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

List focused checks for correctness separately from measurements for a performance
claim. For every unavailable check, name its configuration, missing resource and next
feasible action. Follow [scope-and-validation.md](scope-and-validation.md): unrun FP8
or hardware checks are validation limits, not exclusions. Connect each check to the
example where possible.
````

If the logic is not yet understood, record the exact unresolved contract and continue
source/target analysis before making an adaptation patch. A user-required missing
capability stays `PORT_REQUIRED` with `implementation_status: REVIEWING`; missing
logic is work to resolve, not permission to discard it. Use `NEEDS_REVIEW` when target
applicability or selection is genuinely undecided, preserving any user selection.
An explanatory example does not require hardware, model weights or measured gains.
