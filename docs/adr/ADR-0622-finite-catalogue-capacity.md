# ADR-0622 — Finite replay capacity for a third TRAIN task

## Context

The accepted after-B affine model solves known A/B games. The existing bank
retains one packet per public cue, so its next C_train fit contains B+C and
excludes A. Sequential replay would replace the head again. The existing joint
composer and learner already support27rows; no new model or runner is required.

## Decision

Add an explicit capacity2 option to the same arrived-only FIFO composer, keeping
default1. Record every old source and arrival, fixed source spans, replay
exposure, materialized arrays and finite-bank numeric equivalents. Older scalar
fields alias the most recent prior packet; plural fields contain all prior data.

Use the existing delayed AttributionStream for two C_train-only wakes in each
of two fixed study forks of the accepted after-B model. Do not refit the shared
prefix. Keep native source4/54 counters, separate source6/90 sunk history and
new per-arm/shared work explicit. These are new study forks, not resumed old
consent/publication/budget authority. Local suffix clocks and historical offset
are different namespaces. Original source and actors remain unchanged.

Finish both fixed training arms before any diagnostic scoring, including the
previously unmeasured C_train reference from the original after-B model. This
shows whether C accuracy changes without using scores to choose the protocol.
Use the existing CLI with explicit original TRAIN and retained-model sources.
No final/probe data, live games, promotion, new math or optimization tuning.

## Alternatives

An unlimited bank hides cost and is unnecessary. Scoring only after C cannot
show whether the model already knew C. Repeating A/B prefix fits wastes accepted
work. Reusing the old run's spent budget would misrepresent authority and costs.

## Consequences

Capacity2 adds18suffix replay exposures and up to18bank rows relative to1;
equal update counts are unequal exposure/computation. TRAIN class accuracy and
confidence may disagree. No successful native C or after-C retention, general
reliability, matched-compute advantage or CPC benefit follows from this design.
Keep all negative outcomes and spent earlier failures; freeze the small study
only after software qualification, then record its result without tuning.
