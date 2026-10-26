# ADR-0419: Retain bounded stopping and work through the ordinary stream

## Context

Accepted bounded core and atomic episode updates return richer work, but the
scheduled adapter reconstructed a two-field diagnostic and the cursor accepted
only the exact base type. Native private-candidate publication and outer scheduled
publication have different failure points; combining them would misreport costs.

## Decision

Add one exact frozen categorical subtype of the existing TrainingDiagnostic,
with owned stopping policy/inference, native32work and separate outer7work. Keep
base constructor/serialization unchanged. Admit this exact validated subtype in
ordinary cursors, including complete original fields before copying. Reuse the
existing scheduled candidate/publication path and ordinary delayed stream.

Why: the demonstrated data loss requires richer owned receipts. Separating work
scopes preserves completed candidate work even when the outer update refuses.
Validate known namespace/types/bounds before any success/failure merge; malformed
results cannot overwrite counts or overflow the existing failure port. Prepare
success receipts before publication, then compare actual counters with that
forecast before returning. Partial failures retain actual observed work.

Stopping policy is an explicit optional constructor choice for PC/neutral TRAIN
episodes, copied/validated before allocation and protected by original owner and
immutable scalar/type/float-bit birth values. Fork/deepcopy retain independent
configuration. Native snapshot formats stay unchanged; restores use declared
wrapper configuration and cannot establish historical stopping provenance.

## Alternatives

Adding fields to every base diagnostic would change all old receipt formats.
Dropping richer fields loses actual work. Broad subclass admission would weaken
cursor checks. Rebuilding orchestration or polling is unnecessary: ordinary
pause/refused advance/resume works, while managed deferred polling stays separate.

## Consequences

303scoped tests/statics and independent review accept ordinary stream support.
Success native/outer counts are validated; errors preserve both scopes and flag
unobserved work. Full managed-capture/prune/probe consumers remain separately
unqualified. R2.1 already has isolated probes; next compose explicit reserved-role
bounded learning and complete work propagation there before comparative science.

[Usage and scopes](../categorical-bounded-inference.md#scheduled-train-stream),
[all commands/failures/costs](../../artifacts/runs/categorical-bounded-stream-20261018/RESULTS.md).
