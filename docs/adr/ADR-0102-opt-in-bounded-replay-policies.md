# ADR-0102: Keep bounded replay policies explicit and versioned

## Context

The historical NumPy buffer holds whole batches. The opt-in v4 continual
route already caps unique retained rows and their copied-array bytes, then
keeps the smallest fixed content hashes. P4.1 measured the resulting A→B
survival and cached priorities. Comparing recency with a reservoir-style
sample must not change the historical buffer or v4 checkpoint identity.

## Decision

Add a typed `ReplayRetentionPolicy` for an explicitly configured bounded
core model. `recent_fifo` evicts the oldest retained distinct ID; another
arrival of a retained ID refreshes its row, priority, and recency.
`seeded_reservoir` assigns each distinct content ID a stable SHA-256 rank
from a predeclared 64-bit seed and evicts the largest rank. Both use the
same `ReplayRetentionBudget` example and copied-array byte caps, and both
retain one row per content ID. No task/class label, guard metric, or
final-test value influences retention. Sleep replay selection remains a
separate existing decision.

This is a seeded bottom-k priority reservoir over distinct content IDs, not
classic Algorithm R over every wake occurrence. Under independent uniform
ranks, bottom-k selects a uniform subset of distinct IDs; SHA-256 supplies
a deterministic pseudorandom ranking, not a guarantee about real data
distributions. Repeated epochs reuse an ID's rank and need no unbounded
seen-ID ledger. A new seed is a new declared retention setting, never
chosen from final-test outcomes.

Omitting the policy keeps the prior content-hash algorithm and the same
snapshot field set. Explicit new policies add their identity to the model
snapshot; restore rejects a different policy, seed, or budget before
changing live state. Runner selection, result metadata, duplicate/exposure
reporting, checkpoint versioning, and controlled accuracy comparisons remain
P4.2b.

## Alternatives

- Change the v4 default to FIFO or a seeded sample. That would alter prior
  protocol results and checkpoint interpretation.
- Use classic Algorithm R over repeated wake occurrences. Repeated examples
  would get multiple chances; exact sampling of distinct IDs would require
  an unbounded seen-ID ledger outside the declared retained-data budget.
- Rank rows by model error or task label. Those add separate information and
  policy factors before matched replay controls are available.

## Consequences

Core fixtures compare FIFO and seeded reservoir under identical 4-example /
96-array-byte caps on one fixed A→B stream, test independent count and byte
limits, deduplication, invalid settings, and snapshot continuation. They are
correctness observations, not evidence of a learning benefit. The existing
v4 hash route and legacy batch route remain the defaults for their protocol
IDs. P4.2 stays open for versioned runner artifacts and a predeclared
policy comparison.
