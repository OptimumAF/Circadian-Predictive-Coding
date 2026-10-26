# ADR-0559 — Diagnose paid BP checkpoints before another fit

## Context

#41native families fail complete ticket tasks; conventional BP briefly performs
two of three steps. Actual saved BP checkpoints and typed raw scores already
exist, and the existing core probability-margin reducer can explain TRAIN fit.

## Decision

Use a small offline scalar-evidence script over saved affine tensors and the
original teacher. Verify reference float32 projection against every paid native
score before interpreting phase margins. Keep global/allowed classes and
historical exposure states distinct; no source fit, native restore or new runner.

Why this: attaching diagnostics by rerunning the baseline would spend training
without resolving a different uncertainty. Saved evidence suffices for this fit/
alignment diagnosis and preserves original negative metrics unchanged.

## Alternatives

- Repeat training to attach an observer: unnecessary for the saved fixed head.
- Reconstruct optimizer/gradient/native state: no continuation or updates needed.
- Tune dose/architecture first: obscures the demonstrated phase errors.
- Reuse unrelated game/CLM loaders: incompatible schemas and unnecessary coupling.

## Consequences

Numerical equivalence is measured within a prospective tolerance, not universal
backend identity. Pending TRAIN labels are offline only, unavailable to earlier
controllers. Diagnosis shows insufficient task-action fit despite falling CE;
it does not isolate causal optimization/capacity/interference mechanisms or prove
full-task learning. A bounded isolated conventional TRAIN fit is the next step.
