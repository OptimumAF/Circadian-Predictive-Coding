# ADR-0031: Reject nonfinite circadian configuration at construction

## Context

The NumPy and Torch circadian config validators checked many ranges but
used comparisons such as `value <= 0`. A NaN in a positive-only field
passes that comparison, and an infinity may pass a positive lower bound.
This affected options even when their mechanisms were currently disabled,
leaving a later training or sleep event to encounter the value.

## Decision

At construction, both validators first require every numeric config field
to be finite, with a field-named `ValueError`. This covers the declared
floating hyperparameters, integer counts, and boolean switches when a
caller supplies a nonfinite number in their place. Existing range checks
then retain their original constraints and messages for finite values.
Defaults, matched-control presets, and valid finite arithmetic are
unchanged. The check is construction-only and adds no training-step work.

## Evidence and limits

An exhaustive fixture substitutes NaN, positive infinity, and negative
infinity into each numeric field independently: 67 NumPy and 58 Torch
fields, including disabled mechanisms. The initial case failed the new
field-name/finite error contract; all substitutions now raise as specified.
Default and named matched-control construction still succeeds. See
`tests/test_circadian_finite_config.py` and `docs/development-log.md`.

This validation concerns the two circadian config dataclasses. Dataset,
benchmark protocol, and command-line configuration gates have separate
owners; model constructor dimensions remain P2.8e, while finite step rates
are checked at their training boundaries.
