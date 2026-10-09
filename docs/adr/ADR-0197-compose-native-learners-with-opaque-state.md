# ADR-0197 — Compose native learners with opaque state

## Context

The existing CPC and ordinary-gradient NumPy models expose different diagnostics
and model channels. CPC has a complete public snapshot; BackpropMLP has no equivalent
public port. The existing ToyBudgetSession already checks complete wake boundaries.
Duplicating those controls or assuming a shared loss would obscure the comparison.
The existing ordinary checkpoint validator is tied to two-feature trial prefixes
and expected update counts; this model-state adapter must support other input
widths without changing that historical checkpoint contract.

## Decision

Define a small generic core learner protocol with train/predict/snapshot/restore.
Keep diagnostics identified by their existing native definitions. One app step
reuses ToyBudgetSession; outer NumPy adapters own copies of the existing models.
Reuse CPC snapshots and add a full graph-copy/validated restore for ordinary state.
Reject resource limits that the shared step cannot compose yet. Keep existing
models, checkpoint formats, source closures and scientific protocols unchanged.

## Alternatives

- Parameter-only snapshots omit traffic, aliases and CPC replay/RNG/topology.
- A shared loss or tensor schema would misrepresent these native learners.
- A framework-wide runtime would add actor/permission/promotion decisions before
  their concrete R3 contracts and acceptance tests exist.

## Consequences

The same synchronous step can operate both models and non-array extension ports.
Model state stays separate from adapter-owned training policy. These are trusted
local in-memory boundaries, not disk persistence or source/native attestation.
Actor/shadow concurrency, arrival permissions, promotion, privacy and measured
serving/resource contention remain separate work. Scientific admission stays closed.
