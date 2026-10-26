# ADR-0396: Reduce action margins from existing paid TRAIN scores

## Context

Matched neural attempts retain zero exact TRAIN acquisition. CE and average
accuracy can obscure the few action decisions that keep a sequence incorrect.
Existing probability/class and prior-gradient diagnostics do not include target
probability minus the strongest competing action probability.

## Decision

Add a pure categorical probability-margin reducer, reusing exact float64/native
class-count/hard-target validation. Keep all rows/classes, exact-zero ties, absent
support and first-argmax accuracy separate. Return detached JSON primitives.

Compose it only through an explicit `train_action_margins`/`--train-action-margins`
flag in the existing invariant-history TRAIN checkpoint route. Default false
preserves reports. TRAIN admission precedes model access. Reuse returned paid
probabilities, with no additional native forward or update and no probe/final
access. Preserve raw scores before optional work; charge costs independently and
guard preceding evidence/source state before publishing additional fields.

## Alternatives

- Extra model passes provide no additional information for these margins.
- Replacing CE/accuracy/exact metrics or selecting rare actions/checkpoints would
  change the frozen question and hide other errors.
- Extending binary probe diagnostics would conflate their inclusive threshold and
  soft-label semantics with full categorical first-argmax/hard-label semantics.

## Consequences

Outputs add bounded linear row/class storage when enabled. Copy bytes/reduction
time are separate from model exposure; array temporaries, Python/report/guard
costs remain part of enclosing runtime measurements. Failures retain paid scores
and charges; no foreign state is rolled back. Dependencies point inward from the
existing adapter to pure core validators/reducer; no new framework/dependency.
Margins describe signed prediction gaps, not causal capacity/interference,
calibrated uncertainty, adaptation or circadian advantage.

[Commands, all-row evidence, failures and independent review](../../artifacts/runs/train-action-margins-20261018/RESULTS.md).
