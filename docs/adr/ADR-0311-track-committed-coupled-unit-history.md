# ADR-0311: Track committed coupled unit history explicitly

## Context

R2.1 already measures activation health/rank and isolated fresh-task learning.
Coupled PC/CPC lacked real unit importance/age fields. Chemistry and a global
clock cannot supply the utility/maturity contract needed for R2.2 renewal.
Exact legacy snapshot, codec and managed-capture schemas must remain meaningful.

## Decision

Use opt-in tracked native subclasses with bounded last-hidden IDs, a defined EMA
of absolute batch-aggregate outgoing pre-gate gradients, and successful-update
ages. Small protected publication hooks retain default native assignments and
prepare tracked history before any commit. Full tracked snapshots use explicit
format2, preserve fixed policy and native state, and are refused by unsupported
format1 codecs/capture. Existing owned learner ports reconstruct exact tracked
classes and preserve complete in-memory state. Read-only summaries validate and
copy native history; they do not infer missing state or establish causal utility.

## Alternatives

- Infer importance from chemistry or age from wall time: unsupported meanings.
- Add optional fields to every old instance/snapshot: silently changes exact schemas.
- Ignore history in persistence: loses complete-state continuation semantics.
- Expand every codec now: unnecessary for the native diagnostic increment.

## Consequences

Default models/math/schema remain available; tracked storage is24bytes per unit
plus metadata/copies, with no success-record chain. Replay advances native age.
History validation failures preserve the complete prior state. Unsupported disk
and managed capture refuse explicitly. Existing probe observer composition and
safe renewal remain separate unfinished requirements. Evidence and exact gates:
artifacts/runs/r21-coupled-unit-history-20261016/RESULTS.md.

## Follow-up: observer integration accepted, 2026-10-16

The existing observer now reads tracked history before feature work and restores
complete state into independent disposable probe forks. A fixed scenario on the
existing rehearsal CLI verifies exact retained curves/reports, measures overhead
and keeps original untracked protocols unchanged. R2.1's original bounded
implementation criteria are accepted. This adds no persistence support, renewal
or causal-utility claim. Evidence: r21-tracked-observer-20261016/RESULTS.md.
