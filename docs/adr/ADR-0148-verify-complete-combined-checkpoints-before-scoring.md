# ADR-0148: Verify complete combined checkpoints before scoring

## Context

C7 established a reproducible train-only combined/full-minus-one gate,
including rejected replay, lineage and complete circadian snapshot hashes.
The combined development comparison is still unscored. A parameter-only
checkpoint check would not detect changed RNG, chemistry, memory or clocks
in after-A copies, despite c7 providing these stronger facts.

## Decision

Freeze a separate outer-development protocol retaining every c7 cell and
setting. Reuse its helpers, keep after-A copies, finish and validate all
seed facts against the saved byte-bound result, then globally verify
parameter/width and complete circadian snapshot identities before the
first outer value. Check copies again after scoring. Reuse c4 metrics and
report every full/removal/reference plus predeclared replay/capacity/structure
contrast with costs visible. Bound and independently repeat the public route.

## Alternatives

- Score early seeds as training finishes: weakens the complete gate.
- Retrain after A for its score: creates another training path/state.
- Check only parameter hashes: misses observable snapshot drift.
- Omit duplicate/negative/inactive rows: creates selective reporting.
- Treat development as confirmation or complete growth controls: leaves
  the original matrix and missing parent-selection control unsatisfied.

## Consequences

Holding small model copies costs worker memory, included through serialization
in sampled RSS. No pinned c7/core source changes. Unequal work and capacity
are part of the treatment, not normalized into a winner. C8 does not choose
a confirmation configuration or permit final-role access; c9 and independent
confirmation remain separate unfinished work.
