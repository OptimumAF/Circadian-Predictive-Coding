# ADR-0004: Add Reward-Modulated Wake Updates and Adaptive Sleep Budget Scaling

## Context

External model review highlighted two practical gaps in the circadian learning loop:

- wake learning lacks an explicit task-relevance modulation signal
- sleep structural budgets rely heavily on fixed schedules/hyperparameters

We need incremental improvements that keep the model deterministic, lightweight, and easy to test.

## Decision

Add two optional mechanisms to both circadian implementations:

- `CircadianPredictiveCodingNetwork` (NumPy baseline)
- `CircadianPredictiveCodingHead` (ResNet benchmark path)

1. Reward-modulated wake updates
- Compute batch difficulty from mean absolute output error.
- Compare difficulty against an EMA baseline.
- Scale learning updates by a clipped reward factor.

2. Adaptive sleep budget scaling
- Compute a budget scale from:
  - recent energy plateau severity
  - current hidden chemical variance
- Apply the scale before enforcing configured split/prune limits and fraction caps.

Expose both controls through baseline CLI and ResNet benchmark CLI flags. Keep defaults conservative (`off`) for backward compatibility.

## Alternatives Considered

1. Add RL/Bayesian meta-controller for sleep scheduling.
- Rejected for now: larger complexity and harder reproducibility.

2. Modulate split/prune ranking directly with reward first.
- Deferred: needs clearer attribution of neuron-level contribution and stronger evaluation harness.

3. Apply reward/adaptive budget in ResNet path first.
- Initially rejected for sequencing reasons, then implemented after NumPy validation.

## Consequences

- Circadian wake updates can prioritize harder batches without changing model topology.
- Sleep events become less dependent on manual tuning while staying deterministic.
- New behavior is opt-in, preserving existing experiments by default.
