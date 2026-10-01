# ADR-0151: Verify every parent-control checkpoint before development scoring

Date: 2026-09-30
Status: Accepted

## Context

C9b produced two identical complete unscored eight-cell parent-control
matrices. Selector state affects future proposals but can change without
changing current parameters. A parameter-only check or seed-by-seed scoring
could expose early scores before discovering a late selector mismatch.
The pinned c9b runner returns facts without holding live after-A copies.

## Decision

Freeze c9c's canonical request/result/audit bytes, all cells, primary
metrics, twenty pairs per seed and bounded local costs before new scoring.
Compose existing c9b training/fact helpers in a separate app, retaining
after-A copies before B arrival. Require independently valid and exact
all-seed training facts, then globally match parameter/width and complete
circadian snapshots, including both RNGs and selector/cursor/decision state,
before any outer value. Check again after scoring. Use existing metric/score
arithmetic. Keep all outcomes and unequal costs; preserve final seals and
unused confirmation seeds.

## Alternatives

Parameter-only copies omit future selector state. Scoring each seed before
the complete gate permits partial exposure. Changing pinned c9b to return
live models invalidates its source identity. Retraining for each checkpoint
adds work without improving the boundary.

## Consequences

A new small app/CLI holds copies and assembles the unchanged c9b fact schema.
Whole-worker RSS covers copies; per-arm resource attribution remains open.
C9c is development evidence only and cannot close independent confirmation
or the original matrix requirements. No score selects a treatment.
