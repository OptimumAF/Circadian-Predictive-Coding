# ADR-0147: Compose full/minus-one controls before new growth selection

## Context

The isolated factors have verified train/scoring gates, but the minimum
matrix still requires a combined model, component removals and growth controls.
The existing component sleep path implements all required effects. Its policy
API accepts add counts and explicit prunes while choosing split parents by its
own usage ranking; no scheduled/random parent selector exists.

## Decision

Freeze a separate train-only 17-cell/three-seed combined gate using current
component switches. Enable existing default reward modulation in the full
config to define difficulty removal. Retain equal initialization/exposure,
exact neutral PC controls, full-controller ID-matched replay consumers and
prospectively width-14 references. Bound actual and rejected execution plus
transient capacity. Reuse current guarded sleep and record complete rollback
state fingerprints. Verify and repeat before separately frozen outer scoring.
Keep scheduled/random growth and independent confirmation explicitly open.

## Alternatives

- Call isolated results a completed minimum matrix: leaves combined/growth
  interactions unverified.
- Reuse v14 final scores for new selection: those final roles were released.
- Enable adaptive scheduling and change thresholds until it fires: would
  select a different treatment from observed outcomes.
- Label add-count proposals random growth: parents remain usage-ranked.
- Add a new growth selector before the combined correctness gate: introduces
  an unnecessary simultaneous core change. C9 retains that required control.

## Consequences

The combined gate has intentionally unequal replay/topology/guard work across
removals; costs stay beside facts. Difficulty removal also changes weighted
importance history. Inactive or rejected components remain valid findings.
The gate changes no pinned core or older source identity. C7 alone cannot
close P6.3c/P6.3 or permit confirmation/final access.
