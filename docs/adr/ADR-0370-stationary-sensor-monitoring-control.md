# ADR-0370: Explicit stationary sensor monitoring control

## Context

The existing A4/B4/A4 demonstration calibrates on all first4A batches. Its
pre-shift false-alarm denominator is zero; returned-A windows include carryover.
A stationary control is needed to measure post-calibration monitoring without
changing detector thresholds or manufacturing an adaptation advantage.

## Decision

Add an explicit stationary flag through the existing fixture, monitor, adapter
and CLI. Repeat the exact baseline-A arrays in all12 batches with independent
storage; no added noise/RNG. The monitor counts actual postprefix4..11 windows,
including partial failures. Absent B/return phases report not_applicable recovery
and zero false-alarm denominators/null rates. Default False preserves old calls,
descriptions and behavior. No new runner or diagnostic model calls.

Require monitoring. Existing ordinary/scheduled/replay/incident controls compose;
stationary periodic replay pays2deduplicated rows perboundary rather than4/4/2,
without renewing its original12exposure allowance. Heldout and learnability
protocols retain their original recurring phase semantics and are explicitly
refused with this flag until separately specified. All four original methods,
rates, chemistry and initialization laws stay fixed.

## Alternatives

Counting calibration as false-alarm trials biases measurement. Calling the
return phase stationary ignores carryover. Tuning thresholds or adding selected
noise/data would change the question. Another orchestration framework duplicates
the current stream/observer/runtime boundaries.

## Consequences

The runnable control exposes eight dependent postprefix opportunities, not a
statistical FPR guarantee. No shift means no B/return recovery claim. Native
counts and false alarms need frozen local qualification; old evidence remains
unchanged. Existing offline inspector protocol admission needs a separate small
extension for this new explicit stationary report identity.
