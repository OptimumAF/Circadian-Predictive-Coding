# ADR-0045: Report lineage before and after executed sleep

## Context

`SleepEventResult` reported tensor positions for split and prune actions.
After pruning, those positions could no longer identify a removed unit
from the current model state. P3.4c requires stable identities around
changed-width events, while P3.6 and P3.10 will define richer structural
status and sleep telemetry.

## Decision

Both NumPy and Torch sleep results retain their positional indices and
add optional immutable `lineage_before` and `lineage_after` snapshots.
Executed events populate both snapshots, including consolidation with no
topology change. Skipped/disabled events leave both as `None`; existing
manual construction and consumers remain compatible through defaults.
The snapshots contain active stable IDs, birth-parent references, and
the next ID. A caller can compare them to identify actual births and
removals even if split and prune produce the same net width. A gradual
prune that is only scheduled leaves the active IDs unchanged, despite a
positional prune request in the legacy result field.

Why use full snapshots here: they preserve the identity evidence before
the removed ID disappears, without turning current positional result
fields into a new status taxonomy. A later telemetry pass can derive
explicit proposed, scheduled, and removed records without guessing
ancestry from shifted indices.

## Alternatives and consequences

Mapping only the final positions would lose removed-unit IDs. Adding
separate ID arrays for each action would duplicate status semantics
before P3.6 defines them. Pre/post snapshots cost two width-sized tuple
copies per executed event; Torch also transfers IDs to the host. No
learning tensor, selection rule, metric, or non-executed event changes.
P3.6 still owns explicit change-status telemetry and P3.10 the broader
trigger, budget, guard, and duration record.

## Evidence

`tests/test_sleep_event_lineage.py` first failed on the missing event
field. It checks NumPy net-zero-width split/prune and gradual scheduling,
Torch parent/child removal and width growth, immutable retained event
snapshots, and skipped events on both backends. The development log
records the full quality gate.
