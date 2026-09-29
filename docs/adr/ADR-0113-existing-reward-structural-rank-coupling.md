# ADR-0113: Isolate the existing reward-to-structure coupling before extension

## Context

P4.7 asks for a reward-aware structural-ranking test separate from wake
learning-rate modulation and existing importance weighting. NumPy and
Torch already multiply hidden-output gradient magnitude by the current
reward scale when updating per-neuron importance EMA. Split and prune
scores already mix that EMA with chemical usage and output-weight norm.
Adding another reward term without separating these paths could count
the same supervised-error signal twice.

## Decision

Split P4.7 into a fixed read-only rank audit and a later matched
structural outcome gate. The audit uses the existing backend methods on
the same four-neuron state, two eligible IDs, equal output norms, fixed
chemistry, and one split/prune slot. It compares zero importance score
mix, plain gradient importance, and historical reward-weighted gradient
importance while holding the other score inputs fixed. Two fixed
gradient vectors and valid reward factors 1.0 then 1.5 exercise
nonconstant weighting. A constant-factor control tests the normalization
invariance. No optimizer update, sleep mutation, or final role is used.

## Alternatives

- Add a second reward multiplier to split/prune scores now. The existing
  importance path already carries the signal; no outcome evidence yet
  supports counting it again.
- Compare full training with reward modulation on/off and call a rank
  difference structural. That also changes wake learning rate and often
  model weights, preventing factor attribution.
- Use only one backend's score fixture. Both backends implement this
  coupling, and their candidate ordering must be checked separately.

## Consequences

Both backends' fixed scores agree: historical reward weighting flips
split choice from ID 0 to 1 relative to plain importance, and prune
choice from ID 1 to 0. The no-importance lane separates the effect of
importance itself. A constant reward factor changes EMA magnitude but
not min-max-normalized rank. This establishes a distinct existing
rank signal under controlled conditions, not measured learning benefit.
P4.7b must predeclare matched wake, importance-history, and score-mix
factors with fixed structural caps and sealed outcomes before any new
heuristic is accepted. Exact scores are in
`docs/structural-reward-ranking-audit.md`.
