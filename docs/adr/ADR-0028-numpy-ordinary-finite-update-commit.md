# ADR-0028: Stage finite ordinary NumPy training updates

## Context

The NumPy backprop and ordinary predictive-coding trainers accepted finite
inputs and rates but wrote output parameters before discovering that a later
hidden-weight update overflowed. Ordinary PC also computed its diagnostic
after writing parameters, so a finite latent state with an overflowing
squared residual could leave an updated model and an infinite metric.

## Decision

Both ordinary trainers now check existing model parameters and finite
forward intermediates, compute the pre-update diagnostic, and stage every
gradient, candidate parameter, and traffic value. A nonfinite value raises
`FloatingPointError` before any parameter or traffic commit. Valid candidates
are copied into the existing arrays to preserve the public first-layer
aliases. The original diagnostic formulas and update arithmetic remain in
force; this is a commit boundary, not a new learning rule.

The finite check is intentionally at the training-step boundary. It does
not imply stability for arbitrary rates or constrain feedforward-only
prediction calls. The circadian NumPy path and Torch PC heads need their
own adaptive-state and device-aware transaction designs under P2.8c2.

## Evidence and consequences

Deterministic float64 cases use finite `1e200` input/rate to overflow a
hidden-weight candidate after a finite output candidate; a separate PC
case overflows the squared hidden-error diagnostic while all parameters
remain candidate-finite. Both reject without changing parameters or
traffic. Existing latent and gradient tests verify that finite updates and
observable traffic hooks still behave as before. See
`tests/test_post_update_finite_numpy.py` and the development log for
command results.
