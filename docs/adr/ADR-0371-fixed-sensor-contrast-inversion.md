# ADR-0371: Fixed camera contrast inversion in the existing sensor demo

## Context

The accepted brightness and stationary fixtures have perfect cold accuracy.
Their correct execution does not establish learning a new condition or recovery.
The existing delayed stream, matched learners, finish-only A/B readout and
isolated C probes already provide the required adaptation and measurement paths.

## Decision

Add explicit contrast_inversion=False / --contrast-inversion options. The middle
four batches use 1-originalA pixels with the original semantic object labels;
A and its return remain exact. The paired final B set is 1-original noisy finalA,
with the same original seed27031, labels and 32 rows. This models a negative
camera response: the brightness ordering reverses while object identity does not.
No context identifier is provided to the learner. Stationary and inversion modes
are mutually exclusive. Original brightness/stationary defaults remain unchanged.

Why this: one fixed, parameter-free sensor transformation creates an adaptation
challenge without selecting a condition from model scores. Reuse existing final
admission, receipts, budgets and probes rather than another runner. Freeze exact
settings/metrics/work before model execution; retain negative or censored results.

## Alternatives

More brightness, selected noise levels, longer training or chosen initializations
would add tunable choices. A new runtime or learner architecture is unnecessary.
The original easy fixtures remain valuable compatibility and stationary controls.

## Consequences

This introduces conflicting contrast rules across conditions and only two repeated
training patterns per phase. Eight hidden units and four B updates may fail to
learn; that outcome must remain visible. A single-seed engineering run cannot
establish generalization, statistical FPR control, causal recovery or a circadian
advantage. Final data remains unavailable to training, tuning and probe selection;
probe updates stay in disposable forks. Broader scientific requirements stay open.
