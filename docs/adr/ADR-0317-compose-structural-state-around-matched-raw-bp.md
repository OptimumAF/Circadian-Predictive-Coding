# ADR-0317: Compose structural state around matched raw BP

## Context

R2.6 has a measured scheduled-growth path, but its matched raw BP port pins one
configuration. The native BP mathematics and rooted structural envelope already
exist; a width-changing comparison needs their owned composition.

## Decision

Add a separate structural BP adapter. Each validated rooted state constructs an
owned instance of the unchanged strict BP learner at its current raw config.
Pin the original full config and owner learning rate; preserve complete history
through snapshot/fork/build/restore and publish replacements after preparation.
Reject CPC state and incompatible histories. Allow valid earlier rollback.

Reuse the existing capacity per-arm runner for two new BP controls. Keep the
accepted six-arm descriptor and execution bodies intact, with post-run reuse of
their evidence. Freeze work/data/metrics before execution; no baseline tuning.

## Alternatives

- Relaxing fixed-config restore weakens a working contract for unrelated users.
- A second BP kernel or scaling rule breaks native matched-baseline assumptions.
- Algorithm/provenance tags cannot be established from compatible shared raw
  tensors and are not required for layout-safe owned reconstruction.

## Consequences

BP can train after supported structural state transitions using its original
objective. Snapshot compatibility conveys neither operation provenance nor
runtime authority. The small native controls qualify this baseline prerequisite;
complete mechanism/measurement integration and the full R2.6 comparison remain.
