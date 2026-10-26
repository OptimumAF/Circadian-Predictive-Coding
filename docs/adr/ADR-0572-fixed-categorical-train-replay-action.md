# ADR-0572 — Fixed categorical TRAIN replay through original admission

## Context

Finite adaptation represented replay but executed only wait/wake. Categorical
episode codec, exposure charging and native consolidation already exist.
Continuous rehearsal uses incompatible action data; scheduled rehearsal does
not expose an ordinary standalone shared-gate action.

## Decision

Compose a fixed owned TRAIN bank executor into ResourceSharedRuntime. One finite
ID requests a whole bank batch through the original serving gate. Fork complete
candidate state independently, charge before native work, validate the return
and publish through original consolidation. Observe native wake receipts to
advance one expected state between replays; retain bounded metadata. Add an
actor nonblocking snapshot port while preserving original blocking methods.

## Alternatives

Broadly accepting changed counters admits unobserved model changes. Renewing
budgets or retaining all success models violates original authority or bounded
storage. A new runner/sampler duplicates working boundaries. Native rule changes
or final-score-selected replay amounts are unnecessary for this correctness step.

## Consequences

Separate wake/replay charges and aggregate native work remain visible. Failed,
late and unknown work retain charges/known receipts and stop further native work;
consumed events are not retried. Trusted-local finite bounds do not guarantee
hostile-process isolation or hard preemption. Rate/gate/sleep/combined effects
and full R7.1 remain open. The native case verifies mechanics only, with no replay,
controller efficiency, generalization or CPC benefit claim. Review is self-review.
