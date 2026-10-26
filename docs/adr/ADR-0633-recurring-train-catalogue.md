# ADR-0633: Explicit finite recurring TRAIN catalogue

## Context

The existing ordinary arrived-cue composer admits six packets and retains one
or two per cue. Private rehearsal can jointly fit three prior packets plus a
new one, but those private probe records/models cannot become ordinary TRAIN.
A separately supplied fourth TRAIN context and return need ten arrivals.

## Decision

Add keyword `max_arrivals=6`, accepting only exact6 or explicit10, to the existing
composer. Only10 permits `retained_per_cue=3`;1/2 and all old defaults remain.
Reuse the original role validation, witnesses, arrival loop, public-cue routing,
FIFO selection, spans, arrays and return schema. Check accumulated input-array
bytes before the next allocation against the original16MiB bound.

## Alternatives

Broadly widening defaults would silently alter old protocol admission. A live
replay service/new runner is unnecessary for this finite schedule. Reusing the
private F records as TRAIN would violate isolation. None is adopted.

## Consequences

Planning is bounded to ten nine-row TRAIN packets,36 rows per joint fit and54
bank rows. Cost accounting distinguishes arrival/replay exposure and input,
template and bank memory equivalents. The existing learner128-row cap remains.
Focused pure checks validate causality, isolation, mutation refusal, eviction and
legacy parity; no fitting/native outcome follows from these checks. Ordinary
stream integration and a separately frozen outcome protocol remain necessary.
Single-agent self-review is scoped acceptance, not independent review or release
readiness. Evidence: [results](../../artifacts/runs/recurring-train-catalogue-20261024/RESULTS.md).
