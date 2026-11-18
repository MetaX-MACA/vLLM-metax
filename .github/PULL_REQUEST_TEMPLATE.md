<!-- markdownlint-disable -->

## Overview

<!-- Summarize the change in 1-2 lines. Link relevant issues, pull requests and predecessor changes. -->

## Claims

<!-- State the accomplishments in a few concrete bullets: bug fixes, feature support, performance changes, compatibility improvements, etc. Do not claim results that were not validated. -->

## Validation

### Validation Method

<!-- Briefly state how the change was validated and what behavior the checks cover. Detailed explanation of the PR content belongs in Details. -->

### Validation Evidence

<!-- List exact commands and results. Include `pre-commit run --all-files` and its passing result. Use public, reproducible configuration only; do not include vendor/chip or private environment details. -->

### Pending Verification

<!-- Write "None." when no checks are pending. Otherwise use only "- Pending verification: <affected behavior or claim>". Do not disclose why verification is unavailable or include vendor/chip, private package, proprietary environment, internal path or credential details. -->

## Details

<!-- Provide a detailed explanation of the PR content: what changed, motivation, design decisions, root cause, behavior, tradeoffs, limitations and documentation updates. For ported work, include target-first applicability evidence, why an existing target mechanism does not cover the gap, deliberate source deviations and the complete source-to-target mapping. -->

---

<details>
<summary> Pull Request Checklist </summary>

- [ ] The entire PR description is written in English.
- [ ] I followed the contribution and review rules in `AGENTS.md`.
- [ ] I searched for duplicate or overlapping open/recent changes and documented the result.
- [ ] AI assistance was used during the creation of this PR.
- [ ] `pre-commit run --all-files` passed on the final submitted tree.
- [ ] **Design Fit:** Minimizes impact on existing components, reuses existing functionality and justifies added complexity.
- [ ] **Testing and Validation:** Briefly states the validation method, provides reproducible evidence and marks pending checks as `Pending verification`.
- [ ] **Code Quality and Style:** Keeps code and comments clear and concise and updates relevant documentation and examples.
- [ ] **Pull Request Contents:** Provides a detailed explanation of what changed, why it changed, relevant links, root causes, target applicability and implementation tradeoffs.
- [ ] **Public Disclosure:** Does not disclose why verification is unavailable or expose vendor/chip, private environment, internal path or credential details.
- [ ] **Review Readiness:** The human owner is identified, ported work identifies its provenance, and all pending review risks are explicit rather than presented as merge-ready.
</details>

**BEFORE SUBMITTING, PLEASE READ [`AGENTS.md`](../AGENTS.md)**
