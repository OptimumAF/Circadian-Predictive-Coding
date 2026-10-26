# ADR-0499: Compose arrived head corrections with existing reference rehearsal

## Context

Known B control improves after arrived GN correction, while old A is already
lost. Existing ContinuousStreamRehearsal already implements a TRAIN-only bank,
separate original reference budget and paid consolidation semantics. The explicit
head mode unnecessarily refused that composition.

## Decision

Use the existing rehearsal path for explicit bounded common-MSE BP GN. Size the
existing native recorder to eight wakes plus planned replay events, at most16
scalar records and one latest private model pin. Pin original replay budget,
receipt list/objects and native returns before callbacks; verify exact candidate
continuation and preceding success before additional work. Final pure scalar
seals follow facade callbacks. Report actual GN work for wake and replay, and
reference exposure separately from repeated wake rows.

Why this: a missing adapter composition has a direct useful-learning question;
no additional orchestration or optimizer is needed.

## Alternatives

A separate GN replay runner would duplicate scheduling/authority. Ignoring replay
counter changes or refreshing budgets would lose paid provenance. Retaining all
successful model states would grow storage with work. These were not selected.

## Consequences

Default SGD/no-replay interfaces remain covered. Failed attempts keep original
charges and release no candidate. Focused correctness acceptance does not prove
retention or scientific benefit. One separately frozen known-A recovery question
must account for extra rehearsal computation; final evaluation and probes remain
outside correction authority.
