# ADR-0008: Separate vision guard decisions from outer selection

## Context

`vision_validation_unmatched_v1` removed final-test access from training, but
its validation loader served both repeated epoch stopping/sleep rollback and
outer tuning selection. That reuse can adapt a model to the selection set.

## Decision

The default `vision_guard_separated_unmatched_v2` protocol reserves four
roles: wake training, inner guard, outer validation, and final test. Epoch
stopping and sleep rollback read only the guard. The outer validation score
is computed after each candidate finishes training and is used for tuning
selection. Final test remains unavailable to candidate training and tuning.
The report stores ordered sample-ID hashes for every role. The v1 route
remains explicitly selectable for reproducing the prior corrected behavior.

Synthetic guard examples are independently generated from a distinct seed.
For CIFAR, guard and outer validation are disjoint, deterministic views of
the official training source. Neither receives stochastic augmentation.
Both have labeled storage costs: the defaults are 64 synthetic guard examples
or 1,000 CIFAR guard examples, matching the prior validation counts used for
inner decisions, plus separate outer validation counts.
The guard labels are available before training in this offline benchmark;
their use is not a strict-online continual-learning claim. A full CIFAR
source has fewer wake-training examples after the guard reservation.

## Alternatives considered

- Continue using outer validation for both decisions: retains adaptive
  selection pressure from repeated guard checks.
- Use final test as a guard: would invalidate final testing.
- Use retained training examples as a guard: would avoid a holdout cost but
  would measure training fit rather than held-out sleep damage.

## Consequences and open work

The v2 results are not directly comparable with v1 because data allocation
and the stopping signal changed. Both vision protocols still use unmatched
heads and backbone states. A matched-representation track and strict-online
continual protocol remain open. Historical figures and benchmark files are
untouched.
