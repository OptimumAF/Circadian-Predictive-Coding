# ADR-0461: Collect explicit guard-only robot data

## Context

Actual promotable serving is integrated, but legitimate arrived/disjoint robot
guard data is missing. ContinuousEpisode excludes inner_guard and the existing
teacher only collects TRAIN17/19. Existing promotion controller already owns
evaluation isolation and publication; accepted TRAIN trajectories cannot simply
be relabeled as guard evidence.

## Decision

Allocate31/37 prospectively as guard-only resets, preserve TRAIN17/19 and reserved
probe23/final29, and extend existing continuous records/port/shared teacher loop
and CLI with explicit guard collection. Keep the TRAIN API signature/defaults.
Store seed/arrival metadata; return only validated arrived owned guard records.
Episode training conversion, standardization, learners and stream keep refusing
guard data. Existing evaluator/controller retains complete promotion authority.

## Alternatives

Relabeling accepted TRAIN would invalidate isolation. Reusing probe/final seeds
would consume separate evaluation authority. Broad arbitrary-seed collection
would allow outcome-based selection. Another runner duplicates existing IO/work
accounting. A new promotion implementation duplicates the existing controller.

## Consequences

The port now reserves31/37 against non-guard roles. These new allocations carry
no success or A/B-equivalence assumption; no actual31/37 observation occurs in
this increment. Fake IO verifies collection/isolation and preserved failures.
Arrival metadata/getters are trusted local APIs, not arbitrary-code source
certification. Real collection, guard quality and publication require a separate
frozen small protocol. Negative results and full research/release goals remain.
