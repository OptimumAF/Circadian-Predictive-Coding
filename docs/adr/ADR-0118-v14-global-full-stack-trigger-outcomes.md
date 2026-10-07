# ADR-0118: Score every v14 trigger arm only after a common global seal

## Context

The v14 train-only runner establishes conditional matched replay, guard
behavior, and actual structural/work facts. A held-out comparison still
requires all six arm/seed trials to freeze, share the same final A/B
identities within seed, and pass independent work/role/model preflight
before any model prediction uses a final label. Trigger arms can have
different accepted sleep counts and capacity, so equal wake work alone
is not an equal total-compute claim.

## Decision

Use a separate v14 outcome use case. It reruns the complete train-only
preflight immediately before release, including source hashes, initial
trainable states, exact replay IDs/work, typed guard and structural
events, reconstructed stable neuron lineage, clock/exposure totals, and
capacity caps. It then releases all twelve final A/B roles and checks
their IDs and content hashes across arms within each seed before scoring
the first model. Every cell reports all three methods' A-after-A,
A-after-B, and B-after-B accuracy and clipped binary cross entropy,
signed A forgetting, balanced score, wake/replay work, capacity, and
sleep/guard counts. All three within-seed arm pairings are reported with
signed deltas. There is no validation winner or selected arm.

The scored adapter writes deterministic local JSON to a new path with
exclusive creation and LF bytes. The complete fixed protocol and all
six outcomes are retained even if the adaptive trigger is inactive or
the circadian model underperforms a baseline.

## Alternatives

- Score each arm as soon as it finishes. That lets an incomplete or
  invalid later trial coexist with held-out observations and obscures
  the global gate.
- Infer final identity from the same numeric seed alone. A changed
  source or split can preserve seed while changing final labels.
- Compare circadian only against no-sleep. The PC and backprop replay
  controls are necessary to separate sleep timing from extra exposure.

## Consequences

The fixed two-seed study can show a bounded difference but not a broad
trigger advantage. The current adaptive rule did not attempt sleep in
these trials, so its state and metrics equal no-sleep. Periodic adds
replay and prunes capacity, and its effects vary by method and seed.
Forty final rows per phase and weak ordinary-PC A learning for seed 53
limit inference. No threshold, baseline, seed, metric, or stopping rule
was changed after seeing these results. A new trigger rule would need a
separate prospective hypothesis and matched ablation.
