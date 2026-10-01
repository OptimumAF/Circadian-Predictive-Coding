# ADR-0152: Retain every reserved factor in a joint confirmation scope

Date: 2026-09-30
Status: Accepted

## Context

Six development families cover distinct original matrix rows and reserve
ten seeds each. Gating/replay share a reservation; the other four do not.
Some outcomes are null/negative/inactive. Gating/replay runners score outer
data before complete joint training; other pilot validators accept only
their three development seeds. A new confirmation gate must preserve their
treatments without invoking those scoring runners or editing pinned sources.

## Decision

Retain all factors, every named cell/pair and all fifty distinct reserved
seeds. Use independent final roles only after an all-family/all-seed train
and complete-checkpoint gate. Do not open confirmation outer values. First
verify the saved development evidence and freeze a pure inventory: 560 cells,
13,440 wakes, at most 15,620 executed updates including rejected replay,
under 16,000 updates/600 seconds/observed 512 MiB. The joint memory scope
includes six fact histories and both checkpoints; this cap is prospective,
not a measured single-family or per-arm memory claim. Keep unscored and
scored implementation/gates separate, and require the uncertainty contract
before final scoring.

## Alternatives

Selecting only favorable factors violates the informative/null-result
scope. Pooling different families' absolute outcomes or counting shared
seeds twice creates false replication. Calling the two pilot scoring
runners violates the joint evaluation seal. Modifying old sources loses
verified development provenance. A per-family final release exposes scores
before the complete matrix is validated.

## Consequences

A pure manifest and read-only evidence adapter verify the inventory without
data/models. New confirmation orchestration composes existing training
helpers and uses its own strict larger manifest/fact validator. Complete
resource measurements, uncertainty analysis and actual confirmation/final
results remain unfinished until their gates pass. Existing scientific
settings and results remain unchanged.
