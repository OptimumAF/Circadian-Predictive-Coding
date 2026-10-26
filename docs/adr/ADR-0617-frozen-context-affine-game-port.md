# ADR-0617 — Execute fitted affine heads through the existing game loop

## Context

The qualified two-head affine control classifies every original TRAIN action.
Classification fit alone is insufficient evidence of a working closed-loop
policy. The existing evaluator already supplies public history, raw seven-action
argmax, independent forks, snapshot guards and bounded known-A episodes.

## Decision

Add a readonly owned fitted-model adapter and a narrow evaluator wrapper. Reuse
the exact accepted model report through the existing CLI, with no refit. Keep
immutable coefficient leaves, detach model wrappers/bindings and validate them
before further reads. Permit restoration only to the same fitted binding.

Use a fresh zero-training/replay readout budget and retain prior fit costs
separately. Do not fabricate perceptron update clocks or transfer old runtime
authority. Keep the original evaluator loop and default modes unchanged.

Why this: the missing behavior is executing the learned model, not another
learning algorithm, checkpoint format, source framework or game runner.

## Alternatives

Injecting solver weights into perceptron snapshots would misrepresent history.
Refitting during rollout would mix learning with frozen evaluation. A new game
loop would duplicate ownership, termination and action contracts.

## Consequences

The sole known-A blue/red case succeeds in nine steps per cue. It is a frozen
offline joint fit, not chronological adaptation, unseen-layout generalization
or a circadian comparison. Real inference, copy/validation, source IO and
physics costs are retained. The exact local fit is not a bundled release asset;
portable tests explicitly skip its integration checks when artifacts are absent.
All broader learner/scientific/release requirements remain open.
