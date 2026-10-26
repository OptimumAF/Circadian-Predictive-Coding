# ADR0434: use a separate mistake-based linear TRAIN control

## Context

The original54-row TRAIN catalogue has distinct conflict-free canonical inputs.
One-hot least-squares and the longer original H8 fit miss two A labels. Those
results do not prove linear classification impossible. The accepted BP momentum
readout observes local TRAIN descent and does not explain historical F harm.

## Decision

Implement a separate fixed multiclass perceptron: zero coefficients, appended
intercept, supplied-order first-argmax decisions and unit mistake updates. Give it
complete immutable policy/state/counters and the existing NativeLearner interface.
Bind MiniGrid raw296 histories through a typed invariant wrapper, then select it
explicitly in the existing constructive TRAIN-only stream and saved-source CLI.
Keep all original least-squares/neural recipes and scores unchanged.

Why this: a standard trainable classifier adds a useful simple control without
inferring classification limits from a different regression objective. The
existing stream already provides role admission, budgets, ownership, receipts,
fixed checkpoints and scoring; another runner is unnecessary.

## Alternatives and consequences

Do not change the accepted least-squares solver, extend H8 doses or tune seeds.
This learner adds no dependency, RNG, optimizer or CE-optimization claim. Its
parameters/dtype/native objective/computation differ from H8 methods. Counts,
mistakes, prediction work and array-equivalent payloads are reported separately;
failed native prefixes survive missing receipts. A synthetic full-fit failure
is preserved despite successful runtime checks. Real-source fitting requires a
new small frozen protocol; acquired retention, recovery and scientific benefit
remain open. See [usage and evidence](../perceptron-train-fit.md).
