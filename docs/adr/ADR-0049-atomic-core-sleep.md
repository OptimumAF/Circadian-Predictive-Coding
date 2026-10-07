# ADR-0049: Restore core sleep state after a failed event

## Context

Sleep preflight validates requested structure, but later split, prune,
homeostasis, replay, and reset steps can still fail or create a nonfinite
value. A failed event used to leave partial topology, RNG advancement,
replay updates, and counters in the live model. A later seeded update then
depended on an event that was never accepted.

## Decision

After eligibility and budget checks, each NumPy network or Torch head
copies its P3.3 full model state. The event body runs inside a transaction.
If it raises, the original exception is re-raised after restoring the
snapshot. Before success, both backends validate topology and all
model-owned floating arrays and scalar history. NumPy also validates
stored replay batches and priorities. A nonfinite or structurally invalid
post-state therefore rejects the event and restores its entry state.

Skipped or disabled events return before making a snapshot. A successful
event still uses the same proposal order, metric formulas, and result
fields. NumPy's model-owned replay/RNG and Torch's split generator are
inside their existing snapshots. Runner attempt and rollback telemetry
stay outside this state; guard acceptance recovery is P3.7b.

## Consequences

Executed events now incur one full-state copy and post-state check.
Torch's existing detached post-split proposal preflight is still used,
so an event may copy head tensors twice. This is a correctness cost for
the current small local validation; any future optimization must retain
the same complete restoration and seeded continuation contract.

## Evidence

`tests/test_atomic_sleep_core.py` first failed because an injected
post-replay exception left NumPy RNG advanced. It now covers post-replay
and post-split exceptions, nonfinite state, structural misalignment,
complete snapshot equality, and next seeded sleep/wake continuation on
both backends. The development log records the full quality gate.
