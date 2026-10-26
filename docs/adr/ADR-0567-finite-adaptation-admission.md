# ADR-0567 — Finite adaptation choices at existing native admission

## Context

The learned serving path now publishes qualified native candidate weights, but
its update schedule is scripted. R7 needs bounded adaptation choices and causal
fixed/heuristic references before learned control. The existing shared runtime
already checks serving, pauses, resource availability and permanent native
budgets at each update. A second scheduler would duplicate that authority.

## Decision

Define an immutable finite scalar action catalog and envelope in core. The
application owner configures it once; controller input is only a catalog ID.
Compose no-update/bounded-wake choices with ResourceSharedRuntime's existing
admission, original TRAIN runtime and unchanged inbox. Reject unsupported replay,
rate, gate and sleep effects explicitly, preserving unfinished R7.1 requirements.
Keep bounded counters for opportunities, admissions, attempts, completions,
returned polls and unknown work. Retain no success history or model copies.

Keep the charge/refusal transaction serialized and adjacent so every error path
preserves its prefix. Detached snapshots grant no configuration/reset authority.
Original ToyBudget/sharing limits remain binding. Default train_ready behavior
remains available outside opt-in explicit mode. Public controller input carries
no executable callback, permission change or native data reference.

## Alternatives

Arbitrary controller callbacks could bypass declared effects or inspect labels.
Silently treating unsupported actions as no-op would misstate executed behavior.
A new runner or resettable per-choice budget duplicates and weakens the existing
runtime. Assuming an empty poll is free hides opportunities and controller cost.

## Consequences

The runtime can choose whether/how much to wake without changing native
algorithms or releasing final/probe labels. Deferred and failed work remains
accounted, including completed receipts when a later budget check stops. Native
reconstruction/accuracy/retention/promotion evidence from prior tasks stays
separate. Advanced executors, fair controller-overhead/chronology experiments,
generalization and full scientific/release acceptance remain open. Trusted local
Python objects are not a hostile-code containment boundary.
