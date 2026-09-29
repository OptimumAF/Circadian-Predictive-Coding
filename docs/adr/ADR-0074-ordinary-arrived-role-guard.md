# ADR-0074: Run arrived continual roles with an inner-only sleep guard

## Context

The v5 continual runner seals final tests across all seeds, but its one
validation role is descriptive. The four-role source contract in ADR-0073
defines disjoint identities without observing which roles a runner actually
uses. Changing v5 training rows or format-5 checkpoints would invalidate
their checked results and saved identity.

## Decision

Add an opt-in Python API named `continual_arrived_roles_v6` for one fixed
configuration and predeclared seeds. At each phase arrival, split only its
development source into train, inner guard, and outer selection roles.
Phase B's declared exposure fraction is applied before the split; IDs retain
the selected rows' original source positions. Phase B is constructed after
every Phase A model finishes. The existing model training helpers receive
only the arrived train role. On each attempted circadian sleep, compare
accuracy before and after on that phase's inner guard; restore the complete
model snapshot when accuracy falls beyond the declared tolerance. Outer
selection rows are released at arrival but are never passed to training or
the guard in this increment.

Record source and label release at each runner release point, model update
and guard accesses after those calls occur, guard scores/acceptance, and
the phase information available to each method at its first update. Hold
all seed states until every seed has trained, then release final input and
labels and score. Keep the v6 result identity outside the v1–v5 result and
checkpoint routes. The fixed smoke script reports the role hashes, actual
access counts, guard decisions, task information, and descriptive scores.

## Alternatives

- Reuse v5's single validation split. That would mix guard and later
  selection evidence and change its established report identity.
- Score a final test after each seed. That would expose one seed's labels
  before later seeds finish.
- Pass outer-selection labels to the sleep guard. That would let the guard
  influence the role reserved for setting choice.

## Consequences

V6 changes training role sizes and uses guard rollback, so its metrics are
not v5-equivalent. The runner can audit phase arrival and role consumption,
and a final-label perturbation cannot change any trained state. Its event
ledger records API field release and consumption, while the synthetic data
generator can still allocate final arrays internally. The ordinary path
holds all trained states in memory. There is no v6 checkpoint, outer
candidate selection, or frozen selected setting yet; those are
P1.3c3b2c2b–c3. Its scores remain descriptive, and no result establishes
a circadian advantage or a complete strict-online comparison.

ADR-0075 subsequently added completed-seed v6 checkpoint recovery.
ADR-0076 added intra-seed recovery. Outer setting selection remains open.
