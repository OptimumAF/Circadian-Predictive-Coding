# ADR-0451: Freeze TRAIN observation statistics in the BC policy

## Context

The continuous BC baseline improves offline fit but fails its TRAIN robot context.
Its39 varying input columns have a population-scale ratio about1826, alongside
three constants. Conditioning is a concrete experiment; this observation does
not establish the cause of closed-loop failure.

## Decision

Add an optional explicit TRAIN-fitted population mean/std transform, raw by
default. Own it in full policy/snapshot/fork state. Apply the same frozen values
to all train/predict/diagnostic calls, with no fitting during model use. Give
constants unit scale and retain later deviations. Refuse unrepresentable math.
Keep action units/objective/optimizer and existing ports unchanged.

Why this: test a small input-conditioning change while preserving the original
baseline, role isolation, continuation and matched data/update budgets.

## Alternatives and consequences

Implicit online fitting would change prediction and probe behavior and complicate
state ownership. Dropping constants would hide changes in future contexts. A new
runner or codec is unnecessary. Full transform metadata participates in restore
validation and frozen comparisons. Only the missing optional legacy raw field
defaults to null in the existing strict artifact reader; its old source remains.

425 scoped cases and one matched TRAIN comparison pass independent review.
Standardized256 fits better; both fixed policies still fail the TRAIN task.
This does not qualify strong BC, retention/recovery, adaptation or CPC benefit.
