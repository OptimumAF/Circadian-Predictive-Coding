# ADR-0424: Select original TRAIN rows for prior-gradient readouts

## Context

The completed512 H8 BP case still reverses both original A cue turns at step2.
Existing prior diagnostics average all nine rows and cannot recover row gradients
from saved probabilities. Native checkpoint weights were not persisted.

## Decision

Extend the existing reader with default-None row selection over complete TRAIN
episodes. Admit all current/next causal views before selecting numeric arrays;
retain original IDs, state guards and separate full-admission/native work counts.
Add an explicit fixed Astep2 option to the existing constructive fit mode and
saved-source CLI. Reads follow all fit scores under the same original budget,
using owned checkpoint and result copies. Defaults retain previous behavior.

## Alternatives

Cropping episodes would fabricate history and hide unselected corruption.
Inferring parameter gradients from probabilities is impossible without the
native state. A new reader framework or repeated baseline sweep is unnecessary.
One frozen reconstruction is justified only to obtain the missing state and
must reproduce all earlier fixed-point probabilities before qualification.

## Consequences

The API now answers a narrower mathematical question without changing learner
math or admitting final/probe data. It adds validation, copies and backward work
under the original budget. Negative cosine remains descriptive and does not
establish causal interference or an advantage. The old full-fit negative and
unmet mature-A adaptation premise remain intact.
