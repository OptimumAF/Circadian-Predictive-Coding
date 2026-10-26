# ADR-0485: Declare frozen observation scale floors explicitly

## Context

TRAIN control traces show large standardized input departures under tiny nonzero
fitted scales. Empirical scaling currently has no physical minimum. This is not
evidence of causality or an optimal floor. Existing evidence relies on legacy
three-field metadata and unchanged default computations.

## Decision

Add an exact frozen subtype that retains empirical means/scales/fitted rows and
adds positive declared floors and computed effective=max(scale,floor). Preserve
the original class and serialized shape for omitted floors. Existing TRAIN fit,
four-family stream, private updates and frozen serving own the same transform.
Validate redundant derived metadata and explicitly bind selected native requests.
Saved diagnosis must reconstruct selected divisors rather than dropping fields.

## Alternatives

Silently clamp the existing scales would discard empirical provenance and change
old defaults. Refitting on new observations would change the learned input space
and leak unavailable data. A new preprocessing/orchestration framework is not
needed for one frozen transform option.

## Consequences

Numeric statistics payload doubles from16d to32d bytes when selected; max/metadata
validation/copy overhead is additional to existing dense work. Exactconstants keep
legacy scale1. RGB-range composition and canonical saved diagnosis remain separate
contracts. Correctness acceptance does not imply useful control, a safety limit,
reliability, generalization, circadian advantage or release readiness.
